"""Pin the output of the pre-refactoring model into tests/golden/baseline_*.csv.

Run this BEFORE the component refactoring, against `src/financial_model.py`:

    .venv/bin/python scripts/capture_baseline.py

The CSVs it writes are the contract the component-based model has to reproduce
bit for bit (see tests/test_baseline_equivalence.py). They are written with full
float precision so the comparison can be exact rather than approximate.

The scenarios cover every notebook configuration plus the edge cases the
notebooks never hit (payoff inside the horizon, all-cash purchase, zero
interest, negative ETF return).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.financial_model import Scenario, replace, simulate  # noqa: E402

GOLDEN = ROOT / "tests" / "golden"

# --- notebook/buy_vs_rent.ipynb -------------------------------------------
BUY_VS_RENT_COMMON = dict(
    years=30,
    initial_capital=100_000,
    monthly_net_income=10_000,
    yearly_income_increase_rate=2 / 100,
    monthly_spending=5_000,
    yearly_inflation_rate=2 / 100,
    yearly_apartment_price_increase_rate=3 / 100,
    etf_yearly_return_rate=7 / 100,
)

# --- notebook/sondertilgung_vs_etf.ipynb ----------------------------------
SONDERTILGUNG_BASE = Scenario(
    years=30,
    initial_capital=100_000,
    monthly_net_income=5_000,
    yearly_income_increase_rate=2 / 100,
    monthly_spending=2_000,
    yearly_inflation_rate=2 / 100,
    apartment_price=450_000,
    yearly_apartment_price_increase_rate=3 / 100,
    mortgage_interest_rate=3.8 / 100,
    mortgage_down_payment_rate=10 / 100,
    mortgage_fees_rate=10 / 100,
    mortgage_yearly_repayment_rate=1.5 / 100,
    mortgage_refinancing_years=10,
    etf_yearly_return_rate=7 / 100,
    etf_tax_rate=26.375 / 100,
)


def scenarios() -> dict[str, Scenario]:
    out: dict[str, Scenario] = {}

    out["buy_vs_rent__buy"] = Scenario(
        apartment_price=450_000,
        mortgage_interest_rate=3.8 / 100,
        mortgage_down_payment_rate=10 / 100,
        mortgage_fees_rate=10 / 100,
        mortgage_yearly_repayment_rate=1.5 / 100,
        mortgage_refinancing_years=10,
        **BUY_VS_RENT_COMMON,
    )
    out["buy_vs_rent__rent"] = Scenario(
        monthly_rent=1_490,
        yearly_rent_increase_rate=2 / 100,
        **BUY_VS_RENT_COMMON,
    )

    for pct in (0, 1, 2, 3, 4, 5):
        out[f"sondertilgung__{pct}pct"] = replace(
            SONDERTILGUNG_BASE, sondertilgung_yearly_rate=pct / 100
        )

    out["buy_as_investment__property"] = Scenario(
        years=30,
        initial_capital=100_000,
        monthly_net_income=2_000,
        apartment_price=450_000,
        yearly_apartment_price_increase_rate=3 / 100,
        monthly_rental_income=1_200,
        yearly_rental_income_increase_rate=2 / 100,
        mortgage_interest_rate=3.8 / 100,
        mortgage_down_payment_rate=10 / 100,
        mortgage_fees_rate=10 / 100,
        mortgage_yearly_repayment_rate=1.5 / 100,
        mortgage_refinancing_years=10,
        etf_yearly_return_rate=7 / 100,
    )
    out["buy_as_investment__etf_only"] = Scenario(
        years=30,
        initial_capital=100_000,
        monthly_net_income=2_000,
        etf_yearly_return_rate=7 / 100,
    )

    out["etf_withdrawal"] = Scenario(
        years=15,
        initial_capital=130_000,
        monthly_net_income=2_500,
        etf_yearly_return_rate=7 / 100,
        etf_yearly_withdrawal_rate=4 / 100,
        etf_tax_rate=26.375 / 100,
    )

    # Every feature at once (mirrors test_all_factors_together_smoke).
    out["all_factors"] = Scenario(
        years=30,
        initial_capital=150_000,
        monthly_net_income=7_000,
        yearly_income_increase_rate=2 / 100,
        monthly_spending=2_500,
        yearly_inflation_rate=2 / 100,
        monthly_rent=800,
        yearly_rent_increase_rate=1.5 / 100,
        apartment_price=400_000,
        yearly_apartment_price_increase_rate=3 / 100,
        monthly_rental_income=1_300,
        yearly_rental_income_increase_rate=2 / 100,
        mortgage_interest_rate=3.8 / 100,
        mortgage_down_payment_rate=15 / 100,
        mortgage_fees_rate=10 / 100,
        mortgage_yearly_repayment_rate=2 / 100,
        mortgage_refinancing_years=10,
        sondertilgung_yearly_rate=3 / 100,
        etf_yearly_return_rate=7 / 100,
        etf_tax_rate=26.375 / 100,
        etf_yearly_withdrawal_rate=1 / 100,
    )

    # --- edge cases the notebooks never reach ---
    # Loan repaid well inside the horizon: exercises the payment stopping, the
    # unused Sondertilgung reserve flowing into the ETF, and the post-payoff
    # cash flow.
    out["edge__payoff_inside_horizon"] = Scenario(
        years=20,
        initial_capital=100_000,
        monthly_net_income=6_000,
        monthly_spending=1_000,
        yearly_inflation_rate=2 / 100,
        apartment_price=200_000,
        yearly_apartment_price_increase_rate=2 / 100,
        mortgage_interest_rate=3.5 / 100,
        mortgage_down_payment_rate=20 / 100,
        mortgage_fees_rate=10 / 100,
        mortgage_yearly_repayment_rate=4 / 100,
        mortgage_refinancing_years=10,
        sondertilgung_yearly_rate=5 / 100,
        etf_yearly_return_rate=6 / 100,
        etf_tax_rate=26.375 / 100,
    )
    # All-cash purchase: no loan at all, fees still sunk.
    out["edge__full_cash_purchase"] = Scenario(
        years=10,
        initial_capital=600_000,
        monthly_net_income=3_000,
        apartment_price=450_000,
        yearly_apartment_price_increase_rate=3 / 100,
        mortgage_down_payment_rate=1.0,
        mortgage_fees_rate=10 / 100,
        etf_yearly_return_rate=5 / 100,
    )
    # Zero-interest mortgage: linear amortization, no refinancing.
    out["edge__zero_interest_mortgage"] = Scenario(
        years=25,
        initial_capital=120_000,
        monthly_net_income=4_000,
        monthly_spending=1_500,
        apartment_price=300_000,
        mortgage_down_payment_rate=20 / 100,
        mortgage_fees_rate=10 / 100,
        mortgage_yearly_repayment_rate=4 / 100,
        etf_yearly_return_rate=4 / 100,
    )
    # Negative ETF return with a withdrawal on top: capital and cost basis
    # diverge downwards, tax must stay at zero.
    out["edge__negative_etf_return"] = Scenario(
        years=12,
        initial_capital=200_000,
        monthly_net_income=1_000,
        monthly_spending=3_000,
        yearly_inflation_rate=3 / 100,
        etf_yearly_return_rate=-2 / 100,
        etf_tax_rate=26.375 / 100,
        etf_yearly_withdrawal_rate=3 / 100,
    )
    return out


def main() -> None:
    GOLDEN.mkdir(parents=True, exist_ok=True)
    for name, scenario in scenarios().items():
        df = simulate(scenario)
        path = GOLDEN / f"baseline_{name}.csv"
        df.to_csv(path, index=False, float_format="%.17g")
        print(f"{path.relative_to(ROOT)}: {len(df)} rows x {len(df.columns)} cols")


if __name__ == "__main__":
    main()
