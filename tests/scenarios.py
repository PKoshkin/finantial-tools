"""The scenarios the model is pinned against, expressed as component models.

Each builder here mirrors one entry of `scripts/capture_baseline.py`, which
pinned the same configuration through the pre-refactoring `Scenario` model into
`tests/golden/baseline_*.csv`. Keeping the two lists in step is the whole point
of the baseline suite.
"""

from src.model import (
    Etf,
    FinancialModel,
    Home,
    InvestmentProperty,
    Mortgage,
    Rent,
    Salary,
    Spending,
)

NOTEBOOK_MORTGAGE = dict(
    interest_rate=3.8 / 100,
    down_payment_rate=10 / 100,
    fees_rate=10 / 100,
    yearly_repayment_rate=1.5 / 100,
    refinancing_years=10,
)


def buy_vs_rent__buy() -> FinancialModel:
    return FinancialModel(
        components=[
            Salary(monthly_amount=10_000, yearly_increase_rate=2 / 100),
            Spending(monthly_amount=5_000),
            Home(
                price=450_000,
                yearly_price_increase_rate=3 / 100,
                mortgage=Mortgage(**NOTEBOOK_MORTGAGE),
            ),
            Etf(initial_capital=100_000, yearly_return_rate=7 / 100),
        ],
        years=30,
        yearly_inflation_rate=2 / 100,
    )


def buy_vs_rent__rent() -> FinancialModel:
    return FinancialModel(
        components=[
            Salary(monthly_amount=10_000, yearly_increase_rate=2 / 100),
            Spending(monthly_amount=5_000),
            Rent(monthly_amount=1_490, yearly_increase_rate=2 / 100),
            Etf(initial_capital=100_000, yearly_return_rate=7 / 100),
        ],
        years=30,
        yearly_inflation_rate=2 / 100,
    )


def sondertilgung(rate: float) -> FinancialModel:
    return FinancialModel(
        components=[
            Salary(monthly_amount=5_000, yearly_increase_rate=2 / 100),
            Spending(monthly_amount=2_000),
            Home(
                price=450_000,
                yearly_price_increase_rate=3 / 100,
                mortgage=Mortgage(sondertilgung_yearly_rate=rate, **NOTEBOOK_MORTGAGE),
            ),
            Etf(
                initial_capital=100_000,
                yearly_return_rate=7 / 100,
                tax_rate=26.375 / 100,
            ),
        ],
        years=30,
        yearly_inflation_rate=2 / 100,
    )


def buy_as_investment__property() -> FinancialModel:
    return FinancialModel(
        components=[
            Salary(monthly_amount=2_000),
            InvestmentProperty(
                price=450_000,
                yearly_price_increase_rate=3 / 100,
                monthly_rental_income=1_200,
                yearly_rental_income_increase_rate=2 / 100,
                mortgage=Mortgage(**NOTEBOOK_MORTGAGE),
            ),
            Etf(initial_capital=100_000, yearly_return_rate=7 / 100),
        ],
        years=30,
    )


def buy_as_investment__etf_only() -> FinancialModel:
    return FinancialModel(
        components=[
            Salary(monthly_amount=2_000),
            Etf(initial_capital=100_000, yearly_return_rate=7 / 100),
        ],
        years=30,
    )


def etf_withdrawal() -> FinancialModel:
    return FinancialModel(
        components=[
            Salary(monthly_amount=2_500),
            Etf(
                initial_capital=130_000,
                yearly_return_rate=7 / 100,
                tax_rate=26.375 / 100,
                yearly_withdrawal_rate=4 / 100,
            ),
        ],
        years=15,
    )


def all_factors() -> FinancialModel:
    """Every feature switched on at once."""
    return FinancialModel(
        components=[
            Salary(monthly_amount=7_000, yearly_increase_rate=2 / 100),
            Spending(monthly_amount=2_500),
            Rent(monthly_amount=800, yearly_increase_rate=1.5 / 100),
            InvestmentProperty(
                price=400_000,
                yearly_price_increase_rate=3 / 100,
                monthly_rental_income=1_300,
                yearly_rental_income_increase_rate=2 / 100,
                mortgage=Mortgage(
                    interest_rate=3.8 / 100,
                    down_payment_rate=15 / 100,
                    fees_rate=10 / 100,
                    yearly_repayment_rate=2 / 100,
                    refinancing_years=10,
                    sondertilgung_yearly_rate=3 / 100,
                ),
            ),
            Etf(
                initial_capital=150_000,
                yearly_return_rate=7 / 100,
                tax_rate=26.375 / 100,
                yearly_withdrawal_rate=1 / 100,
            ),
        ],
        years=30,
        yearly_inflation_rate=2 / 100,
    )


def edge__payoff_inside_horizon() -> FinancialModel:
    """The loan is repaid long before the horizon ends."""
    return FinancialModel(
        components=[
            Salary(monthly_amount=6_000),
            Spending(monthly_amount=1_000),
            Home(
                price=200_000,
                yearly_price_increase_rate=2 / 100,
                mortgage=Mortgage(
                    interest_rate=3.5 / 100,
                    down_payment_rate=20 / 100,
                    fees_rate=10 / 100,
                    yearly_repayment_rate=4 / 100,
                    refinancing_years=10,
                    sondertilgung_yearly_rate=5 / 100,
                ),
            ),
            Etf(
                initial_capital=100_000,
                yearly_return_rate=6 / 100,
                tax_rate=26.375 / 100,
            ),
        ],
        years=20,
        yearly_inflation_rate=2 / 100,
    )


def edge__full_cash_purchase() -> FinancialModel:
    """Bought outright — no loan, but the transaction costs are still sunk."""
    return FinancialModel(
        components=[
            Salary(monthly_amount=3_000),
            Home(
                price=450_000,
                yearly_price_increase_rate=3 / 100,
                mortgage=Mortgage(down_payment_rate=1.0, fees_rate=10 / 100),
            ),
            Etf(initial_capital=600_000, yearly_return_rate=5 / 100),
        ],
        years=10,
    )


def edge__zero_interest_mortgage() -> FinancialModel:
    return FinancialModel(
        components=[
            Salary(monthly_amount=4_000),
            Spending(monthly_amount=1_500),
            Home(
                price=300_000,
                mortgage=Mortgage(
                    down_payment_rate=20 / 100,
                    fees_rate=10 / 100,
                    yearly_repayment_rate=4 / 100,
                ),
            ),
            Etf(initial_capital=120_000, yearly_return_rate=4 / 100),
        ],
        years=25,
    )


def edge__negative_etf_return() -> FinancialModel:
    """Losses plus a withdrawal: no tax must ever be charged."""
    return FinancialModel(
        components=[
            Salary(monthly_amount=1_000),
            Spending(monthly_amount=3_000),
            Etf(
                initial_capital=200_000,
                yearly_return_rate=-2 / 100,
                tax_rate=26.375 / 100,
                yearly_withdrawal_rate=3 / 100,
            ),
        ],
        years=12,
        yearly_inflation_rate=3 / 100,
    )


BASELINE_MODELS = {
    "buy_vs_rent__buy": buy_vs_rent__buy,
    "buy_vs_rent__rent": buy_vs_rent__rent,
    **{f"sondertilgung__{pct}pct": (lambda pct=pct: sondertilgung(pct / 100)) for pct in range(6)},
    "buy_as_investment__property": buy_as_investment__property,
    "buy_as_investment__etf_only": buy_as_investment__etf_only,
    "etf_withdrawal": etf_withdrawal,
    "all_factors": all_factors,
    "edge__payoff_inside_horizon": edge__payoff_inside_horizon,
    "edge__full_cash_purchase": edge__full_cash_purchase,
    "edge__zero_interest_mortgage": edge__zero_interest_mortgage,
    "edge__negative_etf_return": edge__negative_etf_return,
}
