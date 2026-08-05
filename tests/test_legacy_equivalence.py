"""Equivalence with the models that predate the unified engine.

Before the unification the repository contained three separate models
(src/buy_vs_rent.py, src/buy_as_investment.py, src/etf_investment.py). Their
outputs for the exact scenarios used in the notebooks are pinned in
tests/golden/legacy_*.csv. The component model must still reproduce them
wherever the semantics intentionally coincide.

The deviations are deliberate and were introduced when the models were
unified — three of them fix real money leaks:

- Year 0 row: the component model reports the post-purchase state (loan
  outstanding, fees sunk); legacy buy_vs_rent reported a pre-purchase state.
  Years >= 1 are compared.
- invested_capital: legacy buy_vs_rent's year rows started the counter at
  initial_capital - down_payment - fees while its own year-0 row said
  initial_capital. The component model consistently starts at the sink's
  initial capital, so legacy values differ by exactly down payment + fees.
- Sondertilgung payoff year and later: legacy dropped the unused part of the
  reserve (money vanished); the component model returns it to the sink. Years
  before the payoff year are compared.
- buy_as_investment after payoff: legacy kept deducting the annuity payment
  after the loan was repaid (money vanished); the payment now stops. (In the
  notebook scenario the loan is never repaid within 30 years, so the
  comparison covers all years anyway.)
- Cost basis on sales: legacy reduced the basis by the full sale amount, the
  component model reduces it proportionally (average-cost). This only affects
  the after-tax columns of scenarios with negative monthly contributions
  (Sondertilgung 5%).
- etf_investment.py: legacy added the monthly contribution before applying the
  month's growth; the component model applies growth first, consistent with
  the other legacy models. The final capital differs by well under 1%.
"""

from pathlib import Path

import pandas as pd
import pytest
import scenarios
from src.model import Etf, FinancialModel, Salary, payoff_year

GOLD = Path(__file__).parent / "golden"

CORE_PAIRS = [
    "total_loan",
    "etf_capital",
    "etf_capital_after_tax",
    "estimated_total_capital",
    "property_value",
    "property_equity",
    "cumulative_interest_paid",
    "monthly_interest_payment",
    "monthly_loan_repayment",
]


def assert_columns_match(actual, legacy, columns, years, rtol=1e-9, atol=1e-6):
    a = actual.set_index("year")
    b = legacy.set_index("year")
    for column in columns:
        for year in years:
            got, want = a.loc[year, column], b.loc[year, column]
            assert got == pytest.approx(want, rel=rtol, abs=atol), (
                f"{column} mismatch at year {year}: model={got} legacy={want}"
            )


class TestBuyVsRent:
    def test_buy_scenario(self):
        actual = scenarios.buy_vs_rent__buy().run()
        legacy = pd.read_csv(GOLD / "legacy_buy_vs_rent__buy.csv")
        years = range(1, 31)
        assert_columns_match(actual, legacy, CORE_PAIRS, years)
        # invested_capital: constant documented offset of down payment + fees
        offset = 450_000 * 0.2
        a = actual.set_index("year")
        b = legacy.set_index("year")
        for year in years:
            assert a.loc[year, "invested_capital"] - offset == pytest.approx(
                b.loc[year, "invested_capital"], rel=1e-9
            )

    def test_rent_scenario(self):
        actual = scenarios.buy_vs_rent__rent().run()
        legacy = pd.read_csv(GOLD / "legacy_buy_vs_rent__rent.csv")
        assert_columns_match(
            actual, legacy, CORE_PAIRS + ["invested_capital"], range(1, 31)
        )


class TestSondertilgung:
    @pytest.mark.parametrize("pct", [0, 3, 5])
    def test_matches_until_the_legacy_payoff(self, pct):
        actual = scenarios.sondertilgung(pct / 100).run()
        legacy = pd.read_csv(GOLD / f"legacy_sondertilgung_{pct}pct.csv")
        legacy_paid = legacy[(legacy.year > 0) & (legacy.total_loan <= 0)]
        payoff = int(legacy_paid.year.iloc[0]) if not legacy_paid.empty else None
        years = [y for y in range(1, 31) if payoff is None or y < payoff]
        assert_columns_match(
            actual,
            legacy,
            ["total_loan", "etf_capital", "cumulative_interest_paid", "yearly_sondertilgung"],
            years,
        )
        if pct == 0:
            # no reserve -> no negative contributions -> after-tax identical too
            assert_columns_match(
                actual,
                legacy,
                ["etf_capital_after_tax", "estimated_total_capital"],
                range(1, 31),
            )

    def test_unused_reserve_is_reinvested_not_lost(self):
        """The legacy model dropped the unused reserve in the payoff year."""
        actual = scenarios.sondertilgung(5 / 100).run()
        legacy = pd.read_csv(GOLD / "legacy_sondertilgung_5pct.csv")
        payoff = payoff_year(actual)
        assert payoff is not None
        a = actual.set_index("year")
        b = legacy.set_index("year")
        for year in range(payoff, 31):
            assert a.loc[year, "etf_capital"] > b.loc[year, "etf_capital"]


class TestBuyAsInvestment:
    def test_property_scenario(self):
        actual = scenarios.buy_as_investment__property().run()
        legacy = pd.read_csv(GOLD / "legacy_buy_as_investment__property.csv")
        # The loan is never repaid within 30 years here, so the legacy
        # "payment continues after payoff" bug never triggers and every year
        # must match.
        assert payoff_year(actual) is None
        assert_columns_match(
            actual,
            legacy,
            [
                "total_loan",
                "etf_capital",
                "property_value",
                "property_equity",
                "estimated_total_capital",
                "monthly_interest_payment",
            ],
            range(1, 31),
        )

    def test_etf_only_scenario(self):
        actual = scenarios.buy_as_investment__etf_only().run()
        legacy = pd.read_csv(GOLD / "legacy_buy_as_investment__etf_only.csv")
        assert_columns_match(
            actual, legacy, ["etf_capital", "estimated_total_capital"], range(0, 31)
        )


class TestEtfScript:
    def test_close_to_the_legacy_script(self):
        actual = FinancialModel(
            components=[
                Salary(monthly_amount=2_500),
                Etf(
                    initial_capital=130_000,
                    yearly_return_rate=7 / 100,
                    yearly_withdrawal_rate=4 / 100,
                ),
            ],
            years=15,
        ).run()
        legacy_capital, _ = [
            float(value)
            for value in (GOLD / "legacy_etf_investment.txt").read_text().strip().split(",")
        ]
        final = actual.etf_capital.iloc[-1]
        # the legacy script compounded contributions one month early; <1% apart
        assert final == pytest.approx(legacy_capital, rel=0.01)
        assert final < legacy_capital
