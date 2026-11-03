"""The component protocol: hooks, ordering primitives and contexts.

A component is a frozen dataclass holding *only parameters*. Every mutable
simulation variable lives in the state object its `open()` returns. That split
is what makes a model re-runnable, cheap to copy for sweeps, and safe to
compare against a golden run.

The engine calls the hooks below in a fixed order (see `model.py`). The
granularity is deliberate: each hook corresponds to one group of statements in
the pre-refactoring `simulate()`, which is how the arithmetic stays identical
down to the last bit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any, ClassVar, Sequence


class Compounding(Enum):
    """How a yearly rate is turned into a monthly one.

    The two conventions coexist in this model on purpose, because that is how
    the underlying figures are quoted in the real world:

    - `NOMINAL` — the yearly figure is a nominal rate with monthly accrual, so
      one month costs `rate / 12` and the *effective* year is slightly more
      (3.8% nominal accrues to 3.867% effective). A German mortgage's Sollzins
      works exactly like this, which is why `Mortgage` defaults to it.
    - `EFFECTIVE` — the yearly figure already is the effective annual figure,
      so one month is `(1 + rate) ** (1/12) - 1` and twelve of them reproduce it
      exactly. A quoted "7% a year" market return or "2% inflation" is this,
      which is why `Etf` and `Spending` default to it.

    Both members are available everywhere, but the defaults above are the ones
    the pinned fixtures were computed with. Overriding is a deliberate change
    of assumption: `NOMINAL` on a portfolio makes "7% a year" grow at 7.229%,
    and `EFFECTIVE` on a mortgage makes the loan cheaper than the one the bank
    is actually selling.
    """

    NOMINAL = auto()
    EFFECTIVE = auto()

    def monthly_charge(self, balance: float, yearly_rate: float) -> float:
        """Interest accrued on `balance` over one month.

        Written as a charge rather than a rate so the nominal case keeps the
        literal `balance * rate / 12.0` chain — factoring out `rate / 12` first
        changes the result for about a third of realistic inputs.
        """
        if self is Compounding.NOMINAL:
            return balance * yearly_rate / 12.0
        return balance * ((1.0 + yearly_rate) ** (1.0 / 12.0) - 1.0)

    def monthly_rate(self, yearly_rate: float) -> float:
        """The growth rate applied once per month."""
        if self is Compounding.NOMINAL:
            return yearly_rate / 12.0
        return (1.0 + yearly_rate) ** (1.0 / 12.0) - 1.0


class CashPhase(IntEnum):
    """Order in which monthly cash flows are summed.

    The phase is a property of the component *class*, not of the order the user
    listed the components in, so `components=[...]` can be written in any order
    without changing a single digit of the result.
    """

    INCOME = 10
    PENSION = 15
    RENTAL_INCOME = 20
    OPERATING_COSTS = 25
    SPENDING = 30
    RENT = 40
    DEBT_SERVICE = 50


@dataclass(frozen=True, slots=True)
class CashFlow:
    """A signed monthly cash flow: positive is an inflow."""

    phase: CashPhase
    amount: float


@dataclass(frozen=True, slots=True)
class CashDraw:
    """A one-off outflow at t=0, charged against the cash sink.

    Each draw is applied as its own subtraction — merging two draws into one
    changes the result in floating point.
    """

    amount: float
    label: str


class Cardinality(Enum):
    SINGLETON = auto()  # 0..1 per model
    MULTI = auto()  # 0..N per model


@dataclass(frozen=True, slots=True)
class ModelParams:
    """Parameters shared by every component.

    The inflation rate is handed over as a yearly figure: it is the component
    that decides how to compound it (see `Compounding`), not the model.
    """

    years: int
    start_age: int | None
    yearly_inflation_rate: float
    safe_withdrawal_rate: float


@dataclass(frozen=True, slots=True)
class SetupContext:
    params: ModelParams


@dataclass(frozen=True, slots=True)
class MonthContext:
    params: ModelParams
    year: int  # 1..years
    month: int  # 1..12


@dataclass(slots=True)
class YearContext:
    params: ModelParams
    year: int  # 1..years
    released: list[float] = field(default_factory=list)
    taxable_incomes: list[float] = field(default_factory=list)
    basisrente_contributions: list[float] = field(default_factory=list)
    pension_incomes: list[float] = field(default_factory=list)

    def release(self, amount: float) -> None:
        """Hand cash back to the sink at year end (unused Sondertilgung)."""
        self.released.append(amount)

    def add_taxable_income(self, amount: float) -> None:
        """Announce a year's taxable non-wage result, negative for a loss.

        Written in `year_settle` (stage A), consumed by the taxpayer in
        `year_assess` (stage A2) — different stages, so the component list
        order cannot change who sees what.
        """
        self.taxable_incomes.append(amount)

    def add_basisrente_contribution(self, amount: float) -> None:
        """Announce a year's Basisrente contributions (§10 Abs. 1 Nr. 2 b EStG).

        Altersvorsorgeaufwendungen: the taxpayer deducts them at assessment,
        within the §10 Abs. 3 cap they share with the statutory pension
        contributions — never at payroll. Same stages as `add_taxable_income`.
        """
        self.basisrente_contributions.append(amount)

    def add_pension_income(self, amount: float) -> None:
        """Announce a year's taxable Leibrente (§22 Nr. 1 S. 3 a aa EStG).

        Kept apart from `add_taxable_income` because pension income has its
        own Werbungskosten-Pauschbetrag, and so that the household's rental
        column keeps meaning rental income only.
        """
        self.pension_incomes.append(amount)


class FinancialComponent:
    """Base class for everything a model is built from.

    Subclasses are frozen dataclasses. They declare their identity and cash
    phase as class variables and override the handful of hooks they care about;
    every hook has a no-op default.
    """

    default_name: ClassVar[str] = "component"
    cardinality: ClassVar[Cardinality] = Cardinality.MULTI
    report_kind: ClassVar[str] = ""
    absorbs_cash: ClassVar[bool] = False
    absorbs_taxable_income: ClassVar[bool] = False

    # Set by the engine from the dataclass field of the same name, or derived
    # from `default_name` when the user did not pass one.
    name: str | None = None

    @property
    def is_sink(self) -> bool:
        """True for the one component that absorbs the month's leftover."""
        return False

    @property
    def produces_taxable_income(self) -> bool:
        """True when this component publishes anything the taxpayer assesses.

        Rental results, Basisrente contributions, pension income — every
        `ctx.add_*` channel needs a `GermanHousehold` to consume it.
        """
        return False

    # --- t = 0 -----------------------------------------------------------
    def open(self, ctx: SetupContext) -> Any:
        """Create and return the mutable state for one simulation run."""
        raise NotImplementedError

    def initial_draws(self, state: Any) -> Sequence[CashDraw]:
        """One-off outflows at t=0, in the order they must be subtracted."""
        return ()

    def initial_cash(self, state: Any) -> float:
        """Cash this component brings into the model at t=0."""
        return 0.0

    def seal_basis(self, state: Any) -> None:
        """Components that absorb cash: called once every t=0 draw is applied."""

    def report_initial(self, state: Any) -> dict[str, float]:
        """Columns for the year-0 row (state right after the purchase)."""
        return {}

    # --- start of a year --------------------------------------------------
    def year_begin(self, state: Any, ctx: YearContext) -> None:
        """Reset the accumulators that are averaged over the 12 months."""

    # --- inside a month, one sweep over all components per stage ----------
    def month_begin(self, state: Any, ctx: MonthContext) -> None:
        """Stage (a): service debt, arm this month's Sondertilgung reserve."""

    def cash_flows(self, state: Any, ctx: MonthContext) -> Sequence[CashFlow]:
        """Stage (b): signed flows that make up the month's leftover."""
        return ()

    def claim(self, state: Any, ctx: MonthContext) -> float:
        """Stage (c1): cash withheld from the leftover before the sink sees it."""
        return 0.0

    def absorb(self, state: Any, ctx: MonthContext, amount: float) -> None:
        """Stage (c2): sinks only — grow, then take the allocated amount.

        Called every month even when `amount` is 0.0, otherwise a month of
        compounding would silently disappear.
        """

    def month_end(self, state: Any, ctx: MonthContext) -> None:
        """Stage (e): monthly escalation (spending inflation)."""

    # --- end of a year ----------------------------------------------------
    def year_settle(self, state: Any, ctx: YearContext) -> None:
        """Stage (A): apply the Sondertilgung, release what the loan cannot use."""

    def credit_untaxed(self, state: Any, amount: float) -> None:
        """Stage (A'): sinks only — receive released cash, no growth applied."""

    def year_assess(self, state: Any, ctx: YearContext) -> None:
        """Stage (A2): consume the taxable results published in `year_settle`."""

    def year_withdraw(self, state: Any, ctx: YearContext) -> None:
        """Stage (B): the yearly ETF withdrawal."""

    def report(self, state: Any, ctx: YearContext) -> dict[str, float]:
        """Stage (C): columns for the end-of-year row."""
        return {}

    def year_grow(self, state: Any, ctx: YearContext) -> None:
        """Stage (E): yearly escalation — runs *after* the row is emitted."""

    def year_refinance(self, state: Any, ctx: YearContext) -> None:
        """Stage (F): recompute the annuity at the end of a Zinsbindung."""


def check_compounding(value: Any, name: str = "compounding") -> None:
    if not isinstance(value, Compounding):
        raise TypeError(f"{name} must be a Compounding member")


def check_amount(value: Any, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def check_growth_rate(value: Any, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    if value <= -1.0:
        raise ValueError(f"{name} must be > -1.0")
