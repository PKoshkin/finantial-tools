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


class CashPhase(IntEnum):
    """Order in which monthly cash flows are summed.

    The phase is a property of the component *class*, not of the order the user
    listed the components in, so `components=[...]` can be written in any order
    without changing a single digit of the result.
    """

    INCOME = 10
    RENTAL_INCOME = 20
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
    """Parameters shared by every component."""

    years: int
    start_age: int | None
    yearly_inflation_rate: float
    safe_withdrawal_rate: float
    monthly_inflation_rate: float


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

    def release(self, amount: float) -> None:
        """Hand cash back to the sink at year end (unused Sondertilgung)."""
        self.released.append(amount)


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

    # Set by the engine from the dataclass field of the same name, or derived
    # from `default_name` when the user did not pass one.
    name: str | None = None

    @property
    def is_sink(self) -> bool:
        """True for the one component that absorbs the month's leftover."""
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

    def year_withdraw(self, state: Any, ctx: YearContext) -> None:
        """Stage (B): the yearly ETF withdrawal."""

    def report(self, state: Any, ctx: YearContext) -> dict[str, float]:
        """Stage (C): columns for the end-of-year row."""
        return {}

    def year_grow(self, state: Any, ctx: YearContext) -> None:
        """Stage (E): yearly escalation — runs *after* the row is emitted."""

    def year_refinance(self, state: Any, ctx: YearContext) -> None:
        """Stage (F): recompute the annuity at the end of a Zinsbindung."""


def monthly_rate(yearly_rate: float) -> float:
    """Geometric monthly equivalent of a yearly rate.

    Used for ETF returns and inflation. Mortgage interest deliberately does
    *not* go through this — it uses a plain `rate / 12`, as it always has.
    """
    return (1.0 + yearly_rate) ** (1.0 / 12.0) - 1.0


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
