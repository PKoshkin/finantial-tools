"""A Basisrente (Rürup-Rente): a locked, tax-deductible pension contract.

A fondsgebundene Basisrente (§10 Abs. 1 Nr. 2 b EStG) takes a monthly
contribution, invests what the insurer's costs leave of it in funds, and at
Rentenbeginn turns the contract value into a lifelong pension — never into
cash. There is no surrender value, no capital option, nothing to sell, which
is why the component reports into its own `pension` kind and stays out of
`etf_capital` and `estimated_total_capital`.

The cost model is the one a Produktinformationsblatt lists, and the one that
reproduces a real contract's year-by-year values (NÜRNBERGER tariff
NFX3208T, pinned in tests/test_ruerup.py) month by month:

    value = value * (1 + g_m) * (1 - capital_rate / 12)
            + contribution * (1 - contribution_rate)
            - acquisition_total / acquisition_months     # first months only

Growth comes first, then the contribution, which is the convention of
`Etf.absorb`. `yearly_return_rate` is the funds' own unit-price growth —
after the funds' TER, exactly what an `Etf` is given — so only the
insurer's costs are modelled here. At Rentenbeginn the value grows one more
month, without cost, and becomes a pension of
`capital * (factor / 10_000 + bonus_rate / 12)` a month.

The contract runs on its own clock. `elapsed_months` says how far into it
the model starts; contribution month `contribution_months` is the last one,
and the next month is Rentenbeginn (in this tariff the deferral ends with
the last contribution).

Taxes go through the model's `GermanHousehold`, which a Basisrente therefore
requires. The year's contributions are deducted within the §10 Abs. 3 cap,
at assessment only, so their effect arrives with the next July refund. The
pension is taxed in full: from a 2058 Rentenbeginn on the Besteuerungsanteil
is 100%, and an earlier start would need a Rentenfreibetrag frozen in euros,
which is not modelled.

Deliberately out of scope: a Beitragsfreistellung, an early or late
Rentenbeginn (the insurer's factors depend on age), the kursorientiertes
Ablaufmanagement (returns are deterministic here), the survivors' pension
during the Rentengarantiezeit, and health contributions on the pension
(owed only by voluntarily insured members of the statutory scheme).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..component import (
    Cardinality,
    CashFlow,
    CashPhase,
    Compounding,
    FinancialComponent,
    MonthContext,
    SetupContext,
    YearContext,
    check_amount,
    check_compounding,
    check_growth_rate,
)


def _check_count(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int")


@dataclass(frozen=True)
class RuerupCosts:
    """What the insurer charges, as the Produktinformationsblatt lists it.

    `acquisition_rate` is the Abschluss- und Vertriebskosten share of the
    agreed contribution sum, spread evenly over the first
    `acquisition_months` (a certified Basisrente spreads it over at least
    five years, or over the whole term if that is shorter).
    `contribution_rate` is the Verwaltungskosten share of every
    contribution, `capital_rate` the yearly Verwaltungskosten on the
    contract value, charged a twelfth each month.

    The funds' own costs are not here — they are inside the fund return —
    and neither are the payout costs, which the annuity factor already
    contains. A PIB's "Kapitalkostenbelastung" adds the two capital-based
    parts: 0.80% for the funds of the pinned contract is the insurer's 0.40%
    plus roughly 0.4% of fund costs.
    """

    acquisition_rate: float = 0.025
    acquisition_months: int = 60
    contribution_rate: float = 0.045
    capital_rate: float = 0.004

    def __post_init__(self) -> None:
        for name in ("acquisition_rate", "contribution_rate", "capital_rate"):
            value = getattr(self, name)
            check_amount(value, name)
            if value >= 1.0:
                raise ValueError(f"{name} must be within [0, 1)")
        _check_count(self.acquisition_months, "acquisition_months")
        if self.acquisition_months < 1:
            raise ValueError("acquisition_months must be >= 1")


@dataclass(frozen=True)
class RuerupAnnuity:
    """How the contract value becomes a pension at Rentenbeginn.

    `factor` is the monthly pension per 10,000 EUR of capital. The default
    is the guaranteed Rentenfaktor of the pinned tariff — the only one the
    insurer promises. Its 2024 Rechnungsgrundlagen give 24.996271, and the
    teildynamische Bonusrente adds `bonus_rate` of the capital a year on top
    (1.87% declared for 2024), both only as long as the insurer keeps them.
    `yearly_increase_rate` raises the pension on every anniversary of
    Rentenbeginn: zero at the declared 3% Gesamtverzinsung.
    """

    factor: float = 21.24683
    bonus_rate: float = 0.0
    yearly_increase_rate: float = 0.0

    def __post_init__(self) -> None:
        check_amount(self.factor, "factor")
        if self.factor == 0.0:
            raise ValueError("factor must be > 0")
        check_amount(self.bonus_rate, "bonus_rate")
        check_growth_rate(self.yearly_increase_rate, "yearly_increase_rate")

    def monthly_pension(self, capital: float) -> float:
        """The first monthly pension a capital converts into."""
        return capital * (self.factor / 10_000.0 + self.bonus_rate / 12.0)


@dataclass(slots=True)
class RuerupState:
    value: float  # Vertragswert; zero once converted
    month: int  # contract months begun so far
    monthly_rate: float
    pension: float = 0.0
    pension_capital: float = 0.0
    year_contributions: float = 0.0
    year_costs: float = 0.0
    year_pension: float = 0.0
    cumulative_contributions: float = 0.0
    cumulative_costs: float = 0.0


@dataclass(frozen=True)
class RuerupPension(FinancialComponent):
    """A fondsgebundene Basisrente. Any number of them per model.

    The contribution is taken out of the month's leftover before the sink
    sees it, like a fixed `Etf` contribution, so `monthly_invested` counts
    it; the contract value itself never joins `etf_capital`. A rise of the
    contribution is a second `RuerupPension` for the extra amount, starting
    at `elapsed_months=0` and ending in the same month: that is how the
    insurer books an Erhöhung, with its own acquisition cost on its own sum.

    `initial_value` is the Vertragswert at the model start, from the yearly
    statement. Left at None it is rebuilt by running the first
    `elapsed_months` at `yearly_return_rate` — an estimate, and one that a
    sweep over the return rewrites along with the future.
    """

    default_name = "ruerup"
    cardinality = Cardinality.MULTI
    report_kind = "pension"
    absorbs_cash = True

    monthly_contribution: float = 0.0
    contribution_months: int = 0
    yearly_return_rate: float = 0.0
    elapsed_months: int = 0
    initial_value: float | None = None
    costs: RuerupCosts = RuerupCosts()
    annuity: RuerupAnnuity = RuerupAnnuity()
    compounding: Compounding = Compounding.EFFECTIVE
    name: str | None = None

    def __post_init__(self) -> None:
        check_amount(self.monthly_contribution, "monthly_contribution")
        _check_count(self.contribution_months, "contribution_months")
        if self.contribution_months < 1:
            raise ValueError("contribution_months must be >= 1")
        _check_count(self.elapsed_months, "elapsed_months")
        if not 0 <= self.elapsed_months <= self.contribution_months:
            raise ValueError(
                "elapsed_months must be within [0, contribution_months]: "
                "the model has to start before Rentenbeginn"
            )
        if self.initial_value is not None:
            check_amount(self.initial_value, "initial_value")
        check_growth_rate(self.yearly_return_rate, "yearly_return_rate")
        if not isinstance(self.costs, RuerupCosts):
            raise TypeError("costs must be a RuerupCosts")
        if not isinstance(self.annuity, RuerupAnnuity):
            raise TypeError("annuity must be a RuerupAnnuity")
        check_compounding(self.compounding)

    @property
    def produces_taxable_income(self) -> bool:
        return True

    # --- the contract's own arithmetic, handy in notebooks ---
    @property
    def acquisition_total(self) -> float:
        """Abschluss- und Vertriebskosten on the agreed contribution sum."""
        return self.costs.acquisition_rate * self.monthly_contribution * self.contribution_months

    @property
    def monthly_acquisition(self) -> float:
        return self.acquisition_total / self._acquisition_spread()

    def _acquisition_spread(self) -> int:
        return min(self.costs.acquisition_months, self.contribution_months)

    def _step(
        self, value: float, month: int, contribution: float, rate: float
    ) -> tuple[float, float]:
        """One contribution month: the new value and what it cost."""
        grown = value * (1.0 + rate)
        capital_cost = grown * self.costs.capital_rate / 12.0
        contribution_cost = contribution * self.costs.contribution_rate
        acquisition = self.monthly_acquisition if month <= self._acquisition_spread() else 0.0
        value = grown - capital_cost
        value += contribution - contribution_cost - acquisition
        return value, capital_cost + contribution_cost + acquisition

    def contract_values(self) -> list[float]:
        """The value after every contribution month, from the contract start.

        This is what an insurer tabulates as Verlaufswerte: entry `12 * k - 1`
        is the value at the k-th anniversary. Computed from the terms alone —
        `elapsed_months` and `initial_value` play no part.
        """
        rate = self.compounding.monthly_rate(self.yearly_return_rate)
        value = 0.0
        values = []
        for month in range(1, self.contribution_months + 1):
            value, _ = self._step(value, month, self.monthly_contribution, rate)
            values.append(value)
        return values

    def capital_at_pension_start(self) -> float:
        """The capital converted at Rentenbeginn, from the contract start."""
        rate = self.compounding.monthly_rate(self.yearly_return_rate)
        return self.contract_values()[-1] * (1.0 + rate)

    # --- t = 0 ---
    def open(self, ctx: SetupContext) -> RuerupState:
        rate = self.compounding.monthly_rate(self.yearly_return_rate)
        if self.initial_value is None:
            value = 0.0
            for month in range(1, self.elapsed_months + 1):
                value, _ = self._step(value, month, self.monthly_contribution, rate)
        else:
            value = self.initial_value
        return RuerupState(value=value, month=self.elapsed_months, monthly_rate=rate)

    # --- the month ---
    def month_begin(self, state: RuerupState, ctx: MonthContext) -> None:
        # Rentenbeginn converts here, at stage (a), so that the first
        # pension is already paid in stage (b) of the same month.
        state.month += 1
        if state.month <= self.contribution_months:
            return
        if state.month == self.contribution_months + 1:
            state.pension_capital = state.value * (1.0 + state.monthly_rate)
            state.pension = self.annuity.monthly_pension(state.pension_capital)
            state.value = 0.0
        elif (state.month - self.contribution_months - 1) % 12 == 0:
            state.pension *= 1.0 + self.annuity.yearly_increase_rate
        state.year_pension += state.pension

    def cash_flows(self, state: RuerupState, ctx: MonthContext) -> Sequence[CashFlow]:
        if state.month > self.contribution_months:
            return (CashFlow(CashPhase.PENSION, state.pension),)
        return ()

    def claim(self, state: RuerupState, ctx: MonthContext) -> float:
        return self.monthly_contribution if state.month <= self.contribution_months else 0.0

    def absorb(self, state: RuerupState, ctx: MonthContext, amount: float) -> None:
        if state.month > self.contribution_months:
            return
        state.value, costs = self._step(state.value, state.month, amount, state.monthly_rate)
        state.year_contributions += amount
        state.year_costs += costs
        state.cumulative_contributions += amount
        state.cumulative_costs += costs

    # --- year end ---
    def year_begin(self, state: RuerupState, ctx: YearContext) -> None:
        state.year_contributions = 0.0
        state.year_costs = 0.0
        state.year_pension = 0.0

    def year_settle(self, state: RuerupState, ctx: YearContext) -> None:
        if state.year_contributions > 0.0:
            ctx.add_basisrente_contribution(state.year_contributions)
        if state.year_pension > 0.0:
            ctx.add_pension_income(state.year_pension)

    # --- reporting ---
    def report(self, state: RuerupState, ctx: YearContext) -> dict[str, float]:
        return self._report(state)

    def report_initial(self, state: RuerupState) -> dict[str, float]:
        return self._report(state)

    def _report(self, state: RuerupState) -> dict[str, float]:
        # Cumulative figures count from the model start: before it, only
        # the contract value is known (or estimated).
        return {
            "contract_value": state.value,
            "pension_capital": state.pension_capital,
            "monthly_pension": state.pension,
            "yearly_contributions": state.year_contributions,
            "yearly_costs": state.year_costs,
            "yearly_pension": state.year_pension,
            "cumulative_contributions": state.cumulative_contributions,
            "cumulative_costs": state.cumulative_costs,
        }
