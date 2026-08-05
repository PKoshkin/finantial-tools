"""`FinancialModel` — the engine that drives a set of components through time.

The simulation runs at monthly resolution and reports once a year. The order in
which the engine calls the component hooks is the whole contract; it is spelled
out in `run()` and must not be reshuffled, because the forecast is pinned bit
for bit against the pre-refactoring model in tests/test_baseline_equivalence.py.

Two conventions must survive any future edit:

- How a yearly rate becomes a monthly one is a per-component choice, declared
  by `Compounding`. The defaults — `NOMINAL` for a mortgage's Sollzins,
  `EFFECTIVE` for returns and inflation — are what every pinned figure was
  computed with, so changing a default moves the golden fixtures.
- Sums are built as running left-to-right chains, never as
  `sum(inflows) - sum(outflows)`, and one-off draws are subtracted one at a
  time. In floating point those are different numbers.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import fields, is_dataclass, replace
from typing import Any, Iterable, Sequence

import pandas as pd

from .component import (
    Cardinality,
    FinancialComponent,
    ModelParams,
    MonthContext,
    SetupContext,
    YearContext,
    check_amount,
    check_growth_rate,
)
from .reporting import RESERVED_NAMES, ReportSource, build_row, to_frame

_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class FinancialModel:
    """A financial forecast described by a set of components.

    Parameters shared by every component live here (`years`, `start_age`,
    `yearly_inflation_rate`, `safe_withdrawal_rate`); everything else — income,
    spending, rent, property, portfolios — is a component.

    The instance is immutable: `run()` never mutates it, and `replaced()`
    returns a new model, which is what makes sweeps cheap and safe.
    """

    def __init__(
        self,
        components: Sequence[FinancialComponent],
        *,
        years: int,
        start_age: int | None = None,
        yearly_inflation_rate: float = 0.0,
        safe_withdrawal_rate: float = 0.03,
    ) -> None:
        if not isinstance(years, int) or isinstance(years, bool):
            raise TypeError("years must be an int")
        if years <= 0:
            raise ValueError("years must be > 0")
        if start_age is not None:
            if not isinstance(start_age, int) or isinstance(start_age, bool):
                raise TypeError("start_age must be an int or None")
            if start_age < 0:
                raise ValueError("start_age must be >= 0")
        check_growth_rate(yearly_inflation_rate, "yearly_inflation_rate")
        check_amount(safe_withdrawal_rate, "safe_withdrawal_rate")

        components = tuple(components)
        for component in components:
            if not isinstance(component, FinancialComponent):
                raise TypeError(f"{component!r} is not a FinancialComponent")

        self._params = ModelParams(
            years=years,
            start_age=start_age,
            yearly_inflation_rate=yearly_inflation_rate,
            safe_withdrawal_rate=safe_withdrawal_rate,
        )
        self._names = _assign_names(components)
        self._components = components
        self._by_name = dict(zip(self._names, components))
        _check_cardinality(components)
        self._sink_index = _find_sink_index(components, self._names)
        self._check_initial_capital()
        self._result: pd.DataFrame | None = None

    # --- introspection ---------------------------------------------------
    @property
    def years(self) -> int:
        return self._params.years

    @property
    def start_age(self) -> int | None:
        return self._params.start_age

    @property
    def yearly_inflation_rate(self) -> float:
        return self._params.yearly_inflation_rate

    @property
    def safe_withdrawal_rate(self) -> float:
        return self._params.safe_withdrawal_rate

    @property
    def components(self) -> tuple[FinancialComponent, ...]:
        return self._components

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._names)

    def component(self, name: str) -> FinancialComponent:
        try:
            return self._by_name[name]
        except KeyError:
            raise ValueError(
                f"no component named {name!r}; available: {', '.join(self._names)}"
            ) from None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        parts = ", ".join(
            f"{name}={type(component).__name__}"
            for name, component in zip(self._names, self._components)
        )
        return f"FinancialModel(years={self.years}, {parts})"

    # --- building variants -----------------------------------------------
    def replaced(self, path: str = "", /, **changes: Any) -> "FinancialModel":
        """A copy with something changed.

        `path=""` edits the model's own parameters, `path="etf"` edits that
        component, `path="home.mortgage"` walks into a nested value object.
        """
        if path == "":
            params = dict(
                years=self._params.years,
                start_age=self._params.start_age,
                yearly_inflation_rate=self._params.yearly_inflation_rate,
                safe_withdrawal_rate=self._params.safe_withdrawal_rate,
            )
            unknown = set(changes) - set(params)
            if unknown:
                raise ValueError(
                    f"unknown model parameter(s): {', '.join(sorted(unknown))}; "
                    f"available: {', '.join(sorted(params))}"
                )
            params.update(changes)
            return FinancialModel(self._components, **params)

        head, _, rest = path.partition(".")
        target = self.component(head)
        index = self._names.index(head)
        components = list(self._components)
        components[index] = _replace_at(
            target, rest.split(".") if rest else [], changes, path
        )
        return FinancialModel(
            components,
            years=self._params.years,
            start_age=self._params.start_age,
            yearly_inflation_rate=self._params.yearly_inflation_rate,
            safe_withdrawal_rate=self._params.safe_withdrawal_rate,
        )

    def sweep(self, path: str, field: str, values: Iterable[Any]) -> dict[Any, pd.DataFrame]:
        """Run one forecast per value of a single parameter."""
        return {value: self.replaced(path, **{field: value}).run() for value in values}

    # --- the simulation ---------------------------------------------------
    def run(self) -> pd.DataFrame:
        """Run the forecast and return the yearly DataFrame.

        Row 0 is the state immediately after the t=0 purchases; rows 1..years
        are end-of-year states.
        """
        if self._result is None:
            self._result = self._simulate()
        return self._result.copy()

    def _simulate(self) -> pd.DataFrame:
        params = self._params
        components = self._components
        sink = components[self._sink_index]

        # States are held positionally: two identical components (say two
        # Salary(3_000)) are equal as frozen dataclasses and would collapse
        # into one entry in a dict keyed by the component itself.
        states = [component.open(SetupContext(params)) for component in components]
        sink_state = states[self._sink_index]

        # t=0: the purchases are paid for out of the sink, one draw at a time.
        for component, state in zip(components, states):
            for draw in component.initial_draws(state):
                sink.draw(sink_state, draw.amount)
        for component, state in zip(components, states):
            if component.absorbs_cash:
                component.seal_basis(state)

        invested_capital = 0.0
        for component, state in zip(components, states):
            invested_capital += component.initial_cash(state)

        rows = [
            build_row(
                year=0,
                start_age=params.start_age,
                sources=[
                    ReportSource(name, component.report_kind, component.report_initial(state))
                    for name, component, state in zip(self._names, components, states)
                ],
                invested_capital=invested_capital,
                monthly_leftover=None,
                monthly_invested=None,
                safe_withdrawal_rate=params.safe_withdrawal_rate,
            )
        ]

        for year in range(1, params.years + 1):
            year_ctx = YearContext(params, year)
            for component, state in zip(components, states):
                component.year_begin(state, year_ctx)
            year_leftover = 0.0
            year_invested = 0.0

            for month in range(1, 13):
                month_ctx = MonthContext(params, year, month)

                # (a) service the debt and arm this month's reserve
                for component, state in zip(components, states):
                    component.month_begin(state, month_ctx)

                # (b) the month's cash flows, summed in phase order
                flows = [
                    (flow.phase, index, flow)
                    for index, (component, state) in enumerate(zip(components, states))
                    for flow in component.cash_flows(state, month_ctx)
                ]
                flows.sort(key=lambda item: (item[0], item[1]))
                leftover = 0.0
                for _, _, flow in flows:
                    leftover += flow.amount

                # (c1) what is withheld before the sink sees the money
                remainder = leftover
                claims = []
                for component, state in zip(components, states):
                    claimed = component.claim(state, month_ctx)
                    claims.append(claimed)
                    remainder -= claimed

                # (c2) the portfolios take their share; the sink takes the rest
                invested = remainder
                for index, (component, state) in enumerate(zip(components, states)):
                    if component.absorbs_cash:
                        if index == self._sink_index:
                            component.absorb(state, month_ctx, remainder)
                        else:
                            component.absorb(state, month_ctx, claims[index])
                            invested += claims[index]

                # (d) the leftover is counted *before* the claims, as it always was;
                # `invested` is what actually reached the portfolios, so the two
                # differ by whatever the Sondertilgung reserve withheld.
                year_leftover += leftover
                year_invested += invested
                invested_capital += leftover

                # (e) monthly escalation
                for component, state in zip(components, states):
                    component.month_end(state, month_ctx)

            # (A) extra repayments, and whatever the loans could not use
            for component, state in zip(components, states):
                component.year_settle(state, year_ctx)
            for amount in year_ctx.released:
                if amount > 0:
                    sink.credit_untaxed(sink_state, amount)
                    # It was withheld from the surplus during the year but the
                    # principal could not use it, so it did end up invested.
                    year_invested += amount

            # (B) the yearly withdrawal
            for component, state in zip(components, states):
                component.year_withdraw(state, year_ctx)

            # (C, D) the row
            rows.append(
                build_row(
                    year=year,
                    start_age=params.start_age,
                    sources=[
                        ReportSource(name, component.report_kind, component.report(state, year_ctx))
                        for name, component, state in zip(self._names, components, states)
                    ],
                    invested_capital=invested_capital,
                    monthly_leftover=year_leftover / 12.0,
                    monthly_invested=year_invested / 12.0,
                    safe_withdrawal_rate=params.safe_withdrawal_rate,
                )
            )

            # (E) yearly escalation — after the row, so the row shows the year
            # that was actually lived through
            for component, state in zip(components, states):
                component.year_grow(state, year_ctx)

            # (F) end of a Zinsbindung
            for component, state in zip(components, states):
                component.year_refinance(state, year_ctx)

        return to_frame(rows, start_age=params.start_age)

    # --- validation -------------------------------------------------------
    def _check_initial_capital(self) -> None:
        states = [
            component.open(SetupContext(self._params)) for component in self._components
        ]
        available = sum(
            component.initial_cash(state)
            for component, state in zip(self._components, states)
        )
        required = sum(
            draw.amount
            for component, state in zip(self._components, states)
            for draw in component.initial_draws(state)
        )
        if available < required:
            raise ValueError(
                "initial capital is insufficient to cover down payment and fees "
                f"(available {available:,.2f}, required {required:,.2f})"
            )


def _assign_names(components: Sequence[FinancialComponent]) -> list[str]:
    """Name every component: explicit `name=` wins, the rest get numbered."""
    names: list[str | None] = [component.name for component in components]

    auto: dict[str, list[int]] = defaultdict(list)
    for index, component in enumerate(components):
        if component.name is None:
            auto[component.default_name].append(index)
    for default_name, indices in auto.items():
        if len(indices) == 1:
            names[indices[0]] = default_name
        else:
            for position, index in enumerate(indices, start=1):
                names[index] = f"{default_name}{position}"

    seen: set[str] = set()
    for name in names:
        assert name is not None
        if not _NAME_PATTERN.match(name):
            raise ValueError(
                f"invalid component name {name!r}: use lowercase letters, digits and underscores"
            )
        if name in RESERVED_NAMES:
            raise ValueError(f"component name {name!r} collides with a result column")
        if name in seen:
            raise ValueError(f"duplicate component name {name!r}")
        seen.add(name)
    return names  # type: ignore[return-value]


def _check_cardinality(components: Sequence[FinancialComponent]) -> None:
    counts: dict[type, int] = defaultdict(int)
    for component in components:
        counts[type(component)] += 1
    for component_type, count in counts.items():
        if component_type.cardinality is Cardinality.SINGLETON and count > 1:
            raise ValueError(
                f"at most one {component_type.__name__} component is allowed (got {count})"
            )


def _find_sink_index(
    components: Sequence[FinancialComponent], names: Sequence[str]
) -> int:
    sinks = [
        index
        for index, component in enumerate(components)
        if component.absorbs_cash and component.is_sink
    ]
    if not sinks:
        raise ValueError(
            "the model needs exactly one cash sink: add a portfolio with "
            "monthly_investment=REMAINDER (Etf.cash(...) for plain cash)"
        )
    if len(sinks) > 1:
        offenders = ", ".join(names[index] for index in sinks)
        raise ValueError(
            f"exactly one cash sink is allowed, got {len(sinks)}: {offenders}. "
            "Give all but one a fixed monthly_investment."
        )
    return sinks[0]


def _replace_at(obj: Any, path: list[str], changes: dict[str, Any], full_path: str) -> Any:
    """Rebuild `obj` with `changes` applied at `path` (a chain of attributes)."""
    if not path:
        if not is_dataclass(obj):
            raise ValueError(f"{full_path!r} is not a configurable object")
        known = {field.name for field in fields(obj)}
        unknown = set(changes) - known
        if unknown:
            raise ValueError(
                f"unknown field(s) for {full_path!r}: {', '.join(sorted(unknown))}; "
                f"available: {', '.join(sorted(known))}"
            )
        return replace(obj, **changes)

    head, rest = path[0], path[1:]
    child = getattr(obj, head, None)
    if child is None:
        raise ValueError(f"{full_path!r}: {head!r} is not set on {type(obj).__name__}")
    return replace(obj, **{head: _replace_at(child, rest, changes, full_path)})
