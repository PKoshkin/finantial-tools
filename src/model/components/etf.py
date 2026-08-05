"""The ETF portfolio — the component that holds cash and absorbs the leftover."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..component import (
    Cardinality,
    FinancialComponent,
    MonthContext,
    SetupContext,
    YearContext,
    check_amount,
    check_growth_rate,
    monthly_rate,
)


class _Remainder:
    """Sentinel: this portfolio takes whatever cash the month has left."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "REMAINDER"


REMAINDER = _Remainder()


def sell(capital: float, cost_basis: float, amount: float) -> float:
    """Cost basis after selling `amount` out of a portfolio worth `capital`.

    Average-cost method: the basis shrinks by the sold *share* of the
    portfolio, not by the sold amount. If the portfolio is empty or negative
    the basis is left alone (a degenerate case — the model is in uncovered
    debt).
    """
    if capital <= 0 or amount <= 0:
        return cost_basis
    sold_fraction = min(amount / capital, 1.0)
    return cost_basis * (1.0 - sold_fraction)


@dataclass(slots=True)
class EtfState:
    capital: float
    cost_basis: float
    monthly_rate: float
    withdrawal_gross: float = 0.0
    withdrawal_net: float = 0.0


@dataclass(frozen=True)
class Etf(FinancialComponent):
    """A portfolio that compounds monthly and is taxed on its gains when sold.

    `monthly_investment=REMAINDER` (the default) makes this the model's cash
    sink: every month's surplus flows in and every deficit is covered by
    selling. Exactly one portfolio must be the sink. The capital may go
    negative when the cash flow cannot fund the plan — that deficit keeps
    compounding at the portfolio's rate, i.e. an implicit loan.

    Give `monthly_investment` a number instead to contribute a fixed amount;
    such a portfolio is funded out of the leftover before the sink sees it.
    """

    default_name = "etf"
    cardinality = Cardinality.MULTI
    report_kind = "etf"
    absorbs_cash = True

    initial_capital: float = 0.0
    yearly_return_rate: float = 0.0
    tax_rate: float = 0.0  # capital gains tax, e.g. Abgeltungssteuer
    monthly_investment: Any = REMAINDER
    yearly_withdrawal_rate: float = 0.0
    name: str | None = None

    def __post_init__(self) -> None:
        check_amount(self.initial_capital, "initial_capital")
        check_growth_rate(self.yearly_return_rate, "yearly_return_rate")
        check_amount(self.tax_rate, "tax_rate")
        if not 0.0 <= self.tax_rate < 1.0:
            raise ValueError("tax_rate must be within [0, 1)")
        if not 0.0 <= self.yearly_withdrawal_rate <= 1.0:
            raise ValueError("yearly_withdrawal_rate must be within [0, 1]")
        if self.monthly_investment is not REMAINDER:
            check_amount(self.monthly_investment, "monthly_investment")

    @classmethod
    def cash(cls, initial_capital: float = 0.0, **kwargs: Any) -> "Etf":
        """A zero-return, untaxed portfolio — plain cash sitting in an account."""
        return cls(initial_capital=initial_capital, **kwargs)

    @property
    def is_sink(self) -> bool:
        return self.monthly_investment is REMAINDER

    # --- t = 0 ---
    def open(self, ctx: SetupContext) -> EtfState:
        return EtfState(
            capital=self.initial_capital,
            cost_basis=max(self.initial_capital, 0.0),
            monthly_rate=monthly_rate(self.yearly_return_rate),
        )

    def initial_cash(self, state: EtfState) -> float:
        return self.initial_capital

    def draw(self, state: EtfState, amount: float) -> None:
        """Pay for a one-off purchase out of this portfolio."""
        state.capital -= amount

    def seal_basis(self, state: EtfState) -> None:
        """Fix the cost basis once every t=0 draw has been applied."""
        state.cost_basis = max(state.capital, 0.0)

    # --- the month ---
    def claim(self, state: EtfState, ctx: MonthContext) -> float:
        return 0.0 if self.is_sink else self.monthly_investment

    def absorb(self, state: EtfState, ctx: MonthContext, amount: float) -> None:
        # Growth first, then the contribution: a euro paid in this month does
        # not earn this month's return.
        state.capital = state.capital * (1.0 + state.monthly_rate) + amount
        if amount >= 0:
            state.cost_basis += amount
        else:
            state.cost_basis = sell(state.capital + (-amount), state.cost_basis, -amount)

    # --- year end ---
    def credit_untaxed(self, state: EtfState, amount: float) -> None:
        state.capital += amount
        state.cost_basis += amount

    def year_withdraw(self, state: EtfState, ctx: YearContext) -> None:
        state.withdrawal_gross = 0.0
        state.withdrawal_net = 0.0
        if self.yearly_withdrawal_rate > 0 and state.capital > 0:
            gross = state.capital * self.yearly_withdrawal_rate
            gain_fraction = max(0.0, 1.0 - state.cost_basis / state.capital)
            state.withdrawal_gross = gross
            state.withdrawal_net = gross * (1.0 - gain_fraction * self.tax_rate)
            state.cost_basis = sell(state.capital, state.cost_basis, gross)
            state.capital -= gross

    # --- reporting ---
    def _after_tax(self, state: EtfState) -> float:
        gain = max(0.0, state.capital - state.cost_basis)
        return state.capital - gain * self.tax_rate

    def report(self, state: EtfState, ctx: YearContext) -> dict[str, float]:
        return {
            "capital": state.capital,
            "cost_basis": state.cost_basis,
            "capital_after_tax": self._after_tax(state),
            "yearly_withdrawal_gross": state.withdrawal_gross,
            "yearly_withdrawal_net": state.withdrawal_net,
        }

    def report_initial(self, state: EtfState) -> dict[str, float]:
        return {
            "capital": state.capital,
            "cost_basis": state.cost_basis,
            "capital_after_tax": self._after_tax(state),
            "yearly_withdrawal_gross": 0.0,
            "yearly_withdrawal_net": 0.0,
        }
