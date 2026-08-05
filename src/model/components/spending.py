"""Consumption components: household spending and rent for your own housing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..component import (
    Cardinality,
    CashFlow,
    CashPhase,
    FinancialComponent,
    MonthContext,
    SetupContext,
    YearContext,
    check_amount,
    check_growth_rate,
    monthly_rate,
)


@dataclass(slots=True)
class SpendingState:
    monthly_amount: float
    monthly_rate: float
    year_amount: float = 0.0


@dataclass(frozen=True)
class Spending(FinancialComponent):
    """Household consumption, compounded *monthly*.

    Unlike income and rent this grows every month rather than once a year,
    which is why the reported `monthly_spending` is the 12-month average.
    `yearly_increase_rate=None` means "follow the model's inflation".
    """

    default_name = "spending"
    cardinality = Cardinality.MULTI
    report_kind = "spending"

    monthly_amount: float = 0.0
    yearly_increase_rate: float | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        check_amount(self.monthly_amount, "monthly_amount")
        if self.yearly_increase_rate is not None:
            check_growth_rate(self.yearly_increase_rate, "yearly_increase_rate")

    def open(self, ctx: SetupContext) -> SpendingState:
        rate = (
            ctx.params.monthly_inflation_rate
            if self.yearly_increase_rate is None
            else monthly_rate(self.yearly_increase_rate)
        )
        return SpendingState(monthly_amount=self.monthly_amount, monthly_rate=rate)

    def year_begin(self, state: SpendingState, ctx: YearContext) -> None:
        state.year_amount = 0.0

    def cash_flows(self, state: SpendingState, ctx: MonthContext) -> Sequence[CashFlow]:
        return (CashFlow(CashPhase.SPENDING, -state.monthly_amount),)

    def month_end(self, state: SpendingState, ctx: MonthContext) -> None:
        state.year_amount += state.monthly_amount
        state.monthly_amount *= 1.0 + state.monthly_rate

    def report(self, state: SpendingState, ctx: YearContext) -> dict[str, float]:
        return {"monthly_amount": state.year_amount / 12.0}

    def report_initial(self, state: SpendingState) -> dict[str, float]:
        return {"monthly_amount": state.monthly_amount}


@dataclass(slots=True)
class RentState:
    monthly_amount: float


@dataclass(frozen=True)
class Rent(FinancialComponent):
    """Cold rent you pay for your own housing.

    At most one per model, and it can coexist with a `Home` purchase.
    """

    default_name = "rent"
    cardinality = Cardinality.SINGLETON
    report_kind = "rent"

    monthly_amount: float = 0.0
    yearly_increase_rate: float = 0.0
    name: str | None = None

    def __post_init__(self) -> None:
        check_amount(self.monthly_amount, "monthly_amount")
        check_growth_rate(self.yearly_increase_rate, "yearly_increase_rate")

    def open(self, ctx: SetupContext) -> RentState:
        return RentState(monthly_amount=self.monthly_amount)

    def cash_flows(self, state: RentState, ctx: MonthContext) -> Sequence[CashFlow]:
        return (CashFlow(CashPhase.RENT, -state.monthly_amount),)

    def year_grow(self, state: RentState, ctx: YearContext) -> None:
        state.monthly_amount *= 1.0 + self.yearly_increase_rate

    def report(self, state: RentState, ctx: YearContext) -> dict[str, float]:
        return {"monthly_amount": state.monthly_amount}

    def report_initial(self, state: RentState) -> dict[str, float]:
        return {"monthly_amount": state.monthly_amount}
