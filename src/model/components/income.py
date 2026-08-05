"""Income components."""

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
)


@dataclass(slots=True)
class SalaryState:
    monthly_amount: float


@dataclass(frozen=True)
class Salary(FinancialComponent):
    """Net monthly income, raised once a year.

    The raise happens *after* the year's row is emitted, so the reported
    `monthly_income` is the amount that was actually in force that year.
    """

    default_name = "salary"
    cardinality = Cardinality.MULTI
    report_kind = "salary"

    monthly_amount: float = 0.0
    yearly_increase_rate: float = 0.0
    name: str | None = None

    def __post_init__(self) -> None:
        check_amount(self.monthly_amount, "monthly_amount")
        check_growth_rate(self.yearly_increase_rate, "yearly_increase_rate")

    def open(self, ctx: SetupContext) -> SalaryState:
        return SalaryState(monthly_amount=self.monthly_amount)

    def cash_flows(self, state: SalaryState, ctx: MonthContext) -> Sequence[CashFlow]:
        return (CashFlow(CashPhase.INCOME, state.monthly_amount),)

    def year_grow(self, state: SalaryState, ctx: YearContext) -> None:
        state.monthly_amount *= 1.0 + self.yearly_increase_rate

    def report(self, state: SalaryState, ctx: YearContext) -> dict[str, float]:
        return {"monthly_amount": state.monthly_amount}

    def report_initial(self, state: SalaryState) -> dict[str, float]:
        return {"monthly_amount": state.monthly_amount}
