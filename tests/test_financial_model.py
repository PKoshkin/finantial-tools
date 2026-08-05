"""Tests for the unified financial model.

Two layers:

1. Equivalence with the legacy models. Before the refactoring the repository
   contained three separate models (src/buy_vs_rent.py, src/buy_as_investment.py,
   src/etf_investment.py). Their outputs for the exact scenarios used in the
   notebooks are pinned in tests/golden/*.csv (generated from the legacy code).
   The unified model must reproduce them wherever the semantics intentionally
   coincide. Documented intentional deviations:

   - Year 0 row: unified reports the post-purchase state (loan outstanding,
     fees sunk); legacy buy_vs_rent reported a pre-purchase state. Years >= 1
     are compared.
   - invested_capital: legacy buy_vs_rent's year rows started the counter at
     initial_capital - down_payment - fees while its own year-0 row said
     initial_capital. The unified model consistently starts at initial_capital,
     so legacy values differ by exactly (down_payment + fees).
   - Sondertilgung payoff year and later: legacy dropped the unused part of
     the reserve (money vanished); unified reinvests it into the ETF. Years
     before the payoff year are compared.
   - buy_as_investment after payoff: legacy kept deducting the annuity payment
     after the loan was repaid (money vanished); unified stops the payment.
     (In the notebook scenario the loan is never repaid within 30 years, so
     the comparison covers all years.)
   - ETF cost basis on sales: legacy reduced the basis by the full sale
     amount; unified reduces it proportionally (average-cost method). Affects
     only after-tax columns in scenarios with negative monthly contributions
     (sondertilgung 5%).
   - etf_investment.py script: legacy added the monthly contribution before
     applying the month's growth; unified applies growth first (consistent
     with the other legacy models). Final capital differs by well under 1%.

2. Invariants: cash conservation, closed-form amortization and compounding,
   tax logic, Sondertilgung caps, refinancing, validation.
"""

import math
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from src.financial_model import (
    Scenario,
    calculate_early_repayment_penalty,
    payoff_year,
    simulate,
)

GOLD = Path(__file__).parent / "golden"

BUY_VS_RENT_COMMON = dict(
    years=30,
    initial_capital=100_000,
    monthly_net_income=10_000,
    yearly_income_increase_rate=2 / 100,
    monthly_spending=5_000,
    yearly_inflation_rate=2 / 100,
    yearly_rent_increase_rate=2 / 100,
    yearly_apartment_price_increase_rate=3 / 100,
    mortgage_interest_rate=3.8 / 100,
    mortgage_down_payment_rate=10 / 100,
    mortgage_fees_rate=10 / 100,
    mortgage_yearly_repayment_rate=1.5 / 100,
    mortgage_refinancing_years=10,
    etf_yearly_return_rate=7 / 100,
)

SONDERTILGUNG_COMMON = dict(
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

BUY_AS_INVESTMENT = dict(
    years=30,
    initial_capital=100_000,
    monthly_net_income=2_000,  # legacy "monthly_savings"
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


def assert_columns_match(unified, legacy, pairs, years, rtol=1e-9, atol=1e-6):
    u = unified.set_index("year")
    l = legacy.set_index("year")
    for col_u, col_l in pairs:
        for y in years:
            a = u.loc[y, col_u]
            b = l.loc[y, col_l]
            assert a == pytest.approx(b, rel=rtol, abs=atol), (
                f"{col_u} mismatch at year {y}: unified={a} legacy={b}"
            )


CORE_PAIRS = [
    ("total_loan", "total_loan"),
    ("etf_capital", "etf_capital"),
    ("etf_capital_after_tax", "etf_capital_after_tax"),
    ("estimated_total_capital", "estimated_total_capital"),
    ("property_value", "property_value"),
    ("property_equity", "property_equity"),
    ("cumulative_interest_paid", "cumulative_interest_paid"),
    ("monthly_interest_payment", "monthly_interest_payment"),
    ("monthly_loan_repayment", "monthly_loan_repayment"),
]


# ---------------------------------------------------------------------------
# 1. Equivalence with the legacy models (golden fixtures)
# ---------------------------------------------------------------------------


class TestLegacyEquivalence:
    def test_buy_vs_rent_buy_scenario(self):
        unified = simulate(Scenario(apartment_price=450_000, **BUY_VS_RENT_COMMON))
        legacy = pd.read_csv(GOLD / "legacy_buy_vs_rent__buy.csv")
        years = range(1, 31)
        assert_columns_match(unified, legacy, CORE_PAIRS, years)
        # invested_capital: constant documented offset of down payment + fees
        offset = 450_000 * 0.2
        u = unified.set_index("year")
        l = legacy.set_index("year")
        for y in years:
            assert u.loc[y, "invested_capital"] - offset == pytest.approx(
                l.loc[y, "invested_capital"], rel=1e-9
            )

    def test_buy_vs_rent_rent_scenario(self):
        unified = simulate(
            Scenario(apartment_price=0, monthly_rent=1_490, **BUY_VS_RENT_COMMON)
        )
        legacy = pd.read_csv(GOLD / "legacy_buy_vs_rent__rent.csv")
        pairs = CORE_PAIRS + [("invested_capital", "invested_capital")]
        assert_columns_match(unified, legacy, pairs, range(1, 31))

    @pytest.mark.parametrize("pct", [0, 3, 5])
    def test_sondertilgung_scenarios(self, pct):
        unified = simulate(
            Scenario(sondertilgung_yearly_rate=pct / 100, **SONDERTILGUNG_COMMON)
        )
        legacy = pd.read_csv(GOLD / f"legacy_sondertilgung_{pct}pct.csv")
        legacy_paid = legacy[(legacy.year > 0) & (legacy.total_loan <= 0)]
        payoff = int(legacy_paid.year.iloc[0]) if not legacy_paid.empty else None
        years = [y for y in range(1, 31) if payoff is None or y < payoff]
        gross_pairs = [
            ("total_loan", "total_loan"),
            ("etf_capital", "etf_capital"),
            ("cumulative_interest_paid", "cumulative_interest_paid"),
            ("yearly_sondertilgung", "yearly_sondertilgung"),
        ]
        assert_columns_match(unified, legacy, gross_pairs, years)
        if pct == 0:
            # no reserve -> no negative contributions -> after-tax identical too
            assert_columns_match(
                unified,
                legacy,
                [
                    ("etf_capital_after_tax", "etf_capital_after_tax"),
                    ("estimated_total_capital", "estimated_total_capital"),
                ],
                range(1, 31),
            )

    def test_sondertilgung_payoff_year_reserve_reinvested_not_lost(self):
        """The legacy model dropped the unused reserve in the payoff year."""
        unified = simulate(
            Scenario(sondertilgung_yearly_rate=5 / 100, **SONDERTILGUNG_COMMON)
        )
        legacy = pd.read_csv(GOLD / "legacy_sondertilgung_5pct.csv")
        payoff = payoff_year(unified)
        assert payoff is not None
        u = unified.set_index("year")
        l = legacy.set_index("year")
        # From the payoff year on, unified ETF capital must be strictly higher:
        # the unused reserve is reinvested instead of vanishing.
        for y in range(payoff, 31):
            assert u.loc[y, "etf_capital"] > l.loc[y, "etf_capital"]

    def test_buy_as_investment_property(self):
        unified = simulate(Scenario(**BUY_AS_INVESTMENT))
        legacy = pd.read_csv(GOLD / "legacy_buy_as_investment__property.csv")
        # The loan is never repaid within 30 years in this scenario, so the
        # legacy "payment continues after payoff" bug never triggers and all
        # years must match.
        assert payoff_year(unified) is None
        pairs = [
            ("total_loan", "total_loan"),
            ("etf_capital", "etf_capital"),
            ("property_value", "property_value"),
            ("property_equity", "property_equity"),
            ("estimated_total_capital", "estimated_total_capital"),
            ("monthly_interest_payment", "monthly_interest_payment"),
        ]
        assert_columns_match(unified, legacy, pairs, range(1, 31))

    def test_etf_only(self):
        unified = simulate(
            Scenario(
                years=30,
                initial_capital=100_000,
                monthly_net_income=2_000,
                etf_yearly_return_rate=7 / 100,
            )
        )
        legacy = pd.read_csv(GOLD / "legacy_buy_as_investment__etf_only.csv")
        pairs = [
            ("etf_capital", "etf_capital"),
            ("estimated_total_capital", "estimated_total_capital"),
        ]
        assert_columns_match(unified, legacy, pairs, range(0, 31))

    def test_etf_with_withdrawal_close_to_legacy_script(self):
        unified = simulate(
            Scenario(
                years=15,
                initial_capital=130_000,
                monthly_net_income=2_500,
                etf_yearly_return_rate=7 / 100,
                etf_yearly_withdrawal_rate=4 / 100,
            )
        )
        legacy_cap, _ = [
            float(x)
            for x in (GOLD / "legacy_etf_investment.txt").read_text().strip().split(",")
        ]
        final = unified.etf_capital.iloc[-1]
        # legacy compounded contributions one month early; <1% difference
        assert final == pytest.approx(legacy_cap, rel=0.01)
        assert final < legacy_cap


# ---------------------------------------------------------------------------
# 2. Invariants and closed-form checks
# ---------------------------------------------------------------------------


class TestCashConservation:
    def test_no_money_leaks_with_zero_etf_return(self):
        """With 0% ETF return: every euro of leftover either repays the loan
        (Sondertilgung / annuity) or sits in the ETF. Nothing vanishes."""
        scenario = Scenario(
            sondertilgung_yearly_rate=5 / 100,
            **{**SONDERTILGUNG_COMMON, "etf_yearly_return_rate": 0.0},
        )
        df = simulate(scenario)
        initial_etf = (
            scenario.initial_capital - scenario.down_payment - scenario.purchase_fees
        )
        for _, row in df[df.year > 0].iterrows():
            total_leftover = (
                df[(df.year > 0) & (df.year <= row.year)].monthly_leftover.sum() * 12
            )
            total_sondertilgung = df[
                (df.year > 0) & (df.year <= row.year)
            ].yearly_sondertilgung.sum()
            assert row.etf_capital == pytest.approx(
                initial_etf + total_leftover - total_sondertilgung, abs=1e-6
            )

    def test_leftover_accounting(self):
        """invested_capital = initial_capital + cumulated leftovers."""
        df = simulate(Scenario(apartment_price=450_000, **BUY_VS_RENT_COMMON))
        cum = 100_000.0
        for _, row in df[df.year > 0].iterrows():
            cum += row.monthly_leftover * 12
            assert row.invested_capital == pytest.approx(cum, rel=1e-9)


class TestMortgageMath:
    def test_amortization_matches_closed_form(self):
        """Annuity with monthly interest i/12 and fixed payment P:
        L_n = L0*(1+i/12)^n - P*((1+i/12)^n - 1)/(i/12)."""
        s = Scenario(
            years=5,
            initial_capital=200_000,
            monthly_net_income=10_000,
            apartment_price=450_000,
            mortgage_interest_rate=3.8 / 100,
            mortgage_down_payment_rate=10 / 100,
            mortgage_fees_rate=10 / 100,
            mortgage_yearly_repayment_rate=1.5 / 100,
        )
        df = simulate(s)
        L0 = s.initial_loan
        P = s.initial_monthly_mortgage_payment
        i = s.mortgage_interest_rate / 12.0
        for year in range(1, 6):
            n = 12 * year
            expected = L0 * (1 + i) ** n - P * ((1 + i) ** n - 1) / i
            assert df.set_index("year").loc[year, "total_loan"] == pytest.approx(
                expected, rel=1e-12
            )

    def test_payment_stops_after_payoff_and_cash_flows_to_etf(self):
        """After full repayment the annuity stops and leftover rises by the
        payment amount (this was the buy_as_investment legacy bug)."""
        s = Scenario(
            years=12,
            initial_capital=100_000,
            monthly_net_income=5_000,
            apartment_price=200_000,
            mortgage_interest_rate=3 / 100,
            mortgage_down_payment_rate=30 / 100,
            mortgage_fees_rate=5 / 100,
            mortgage_yearly_repayment_rate=8 / 100,  # aggressive: payoff ~10y
            etf_yearly_return_rate=0.0,
        )
        df = simulate(s).set_index("year")
        payoff = payoff_year(df.reset_index())
        assert payoff is not None and payoff < 12
        after = df.loc[payoff + 1]
        assert after.monthly_interest_payment == 0
        assert after.monthly_loan_repayment == 0
        assert after.monthly_mortgage_payment == 0
        assert after.monthly_leftover == pytest.approx(
            after.monthly_income - after.monthly_spending, rel=1e-9
        )
        # interest stops accumulating
        assert (
            df.loc[payoff + 1 :, "cumulative_interest_paid"].nunique() == 1
        )

    def test_final_payment_capped_at_outstanding(self):
        """The last principal payment never exceeds the remaining loan."""
        s = Scenario(
            years=12,
            initial_capital=100_000,
            monthly_net_income=5_000,
            apartment_price=200_000,
            mortgage_interest_rate=3 / 100,
            mortgage_down_payment_rate=30 / 100,
            mortgage_fees_rate=5 / 100,
            mortgage_yearly_repayment_rate=8 / 100,
        )
        df = simulate(s)
        assert (df.total_loan >= 0).all()

    def test_zero_interest_mortgage_amortizes_linearly(self):
        """With 0% interest the annuity is pure principal: the loan falls by
        loan * repayment_rate per year and no interest is ever paid."""
        s = Scenario(
            years=10,
            initial_capital=100_000,
            monthly_net_income=5_000,
            apartment_price=300_000,
            mortgage_interest_rate=0.0,
            mortgage_down_payment_rate=20 / 100,
            mortgage_fees_rate=10 / 100,
            mortgage_yearly_repayment_rate=5 / 100,
        )
        df = simulate(s).set_index("year")
        loan0 = 240_000
        for y in range(1, 11):
            assert df.loc[y, "total_loan"] == pytest.approx(
                loan0 * (1 - 0.05 * y), abs=1e-6
            )
        assert df.cumulative_interest_paid.iloc[-1] == 0

    def test_full_cash_purchase_no_loan(self):
        """down_payment_rate = 1.0: no loan, no payments; the property price
        plus fees leave the ETF and the equity is the full property value."""
        s = Scenario(
            years=5,
            initial_capital=400_000,
            monthly_net_income=3_000,
            apartment_price=300_000,
            yearly_apartment_price_increase_rate=2 / 100,
            mortgage_down_payment_rate=1.0,
            mortgage_fees_rate=10 / 100,
            etf_yearly_return_rate=5 / 100,
        )
        df = simulate(s).set_index("year")
        assert (df.total_loan == 0).all()
        assert (df.monthly_mortgage_payment == 0).all()
        assert df.loc[0, "etf_capital"] == pytest.approx(400_000 - 330_000)
        for y in range(0, 6):
            assert df.loc[y, "property_equity"] == pytest.approx(
                300_000 * 1.02**y
            )

    def test_refinancing_recomputes_annuity(self):
        """With refinancing the payment drops (lower base), so the loan is
        repaid more slowly than without refinancing."""
        base = dict(
            years=25,
            initial_capital=200_000,
            monthly_net_income=10_000,
            apartment_price=450_000,
            mortgage_interest_rate=3.8 / 100,
            mortgage_down_payment_rate=10 / 100,
            mortgage_fees_rate=10 / 100,
            mortgage_yearly_repayment_rate=2.5 / 100,
        )
        no_refi = simulate(Scenario(**base, mortgage_refinancing_years=0))
        refi = simulate(Scenario(**base, mortgage_refinancing_years=10))
        n = no_refi.set_index("year")
        r = refi.set_index("year")
        for y in range(1, 11):  # identical until the first refinancing
            assert n.loc[y, "total_loan"] == pytest.approx(r.loc[y, "total_loan"])
        for y in range(11, 26):  # afterwards the refinanced loan lags behind
            assert r.loc[y, "total_loan"] >= n.loc[y, "total_loan"] - 1e-6

    def test_sondertilgung_capped_and_reduces_interest(self):
        base = SONDERTILGUNG_COMMON
        st0 = simulate(Scenario(sondertilgung_yearly_rate=0, **base))
        st5 = simulate(Scenario(sondertilgung_yearly_rate=5 / 100, **base))
        max_st = 450_000 * 0.9 * 0.05
        assert (st5.yearly_sondertilgung <= max_st + 1e-6).all()
        assert (
            st5.cumulative_interest_paid.iloc[-1]
            < st0.cumulative_interest_paid.iloc[-1]
        )
        p0, p5 = payoff_year(st0), payoff_year(st5)
        assert p5 is not None and (p0 is None or p5 < p0)


class TestEtfMath:
    def test_etf_only_closed_form(self):
        """FV = P0*(1+g)^n + c*((1+g)^n - 1)/g with monthly rate g."""
        s = Scenario(
            years=10,
            initial_capital=50_000,
            monthly_net_income=1_000,
            etf_yearly_return_rate=6 / 100,
        )
        df = simulate(s)
        g = (1.06) ** (1 / 12) - 1
        n = 120
        expected = 50_000 * (1 + g) ** n + 1_000 * ((1 + g) ** n - 1) / g
        assert df.etf_capital.iloc[-1] == pytest.approx(expected, rel=1e-12)

    def test_tax_applied_only_to_gains(self):
        s = Scenario(
            years=10,
            initial_capital=50_000,
            monthly_net_income=1_000,
            etf_yearly_return_rate=6 / 100,
            etf_tax_rate=26.375 / 100,
        )
        df = simulate(s).set_index("year")
        for y in range(1, 11):
            capital = df.loc[y, "etf_capital"]
            basis = df.loc[y, "etf_cost_basis"]
            expected = capital - max(0.0, capital - basis) * 26.375 / 100
            assert df.loc[y, "etf_capital_after_tax"] == pytest.approx(expected)
            assert df.loc[y, "etf_capital_after_tax"] <= capital

    def test_no_tax_on_losses(self):
        s = Scenario(
            years=5,
            initial_capital=50_000,
            monthly_net_income=1_000,
            etf_yearly_return_rate=-5 / 100,
            etf_tax_rate=26.375 / 100,
        )
        df = simulate(s)
        # capital below basis -> after-tax equals gross
        pd.testing.assert_series_equal(
            df.etf_capital_after_tax, df.etf_capital, check_names=False
        )

    def test_withdrawal_reduces_capital(self):
        """With 0% growth and no contributions the capital shrinks geometrically."""
        s = Scenario(
            years=3,
            initial_capital=100_000,
            etf_yearly_withdrawal_rate=4 / 100,
        )
        df = simulate(s).set_index("year")
        for y in range(1, 4):
            assert df.loc[y, "etf_capital"] == pytest.approx(100_000 * 0.96**y)
            assert df.loc[y, "yearly_etf_withdrawal_gross"] == pytest.approx(
                100_000 * 0.96 ** (y - 1) * 0.04
            )

    def test_withdrawal_tax_on_gain_part_only(self):
        """All-gain portfolio (basis ~ constant, capital grows): the net
        withdrawal must be below gross by roughly gain_fraction * tax."""
        s = Scenario(
            years=2,
            initial_capital=100_000,
            etf_yearly_return_rate=10 / 100,
            etf_yearly_withdrawal_rate=5 / 100,
            etf_tax_rate=25 / 100,
        )
        df = simulate(s).set_index("year")
        y1 = df.loc[1]
        # capital before withdrawal: 110_000, basis 100_000
        gross = 110_000 * 0.05
        gain_fraction = 1 - 100_000 / 110_000
        expected_net = gross * (1 - gain_fraction * 0.25)
        assert y1.yearly_etf_withdrawal_gross == pytest.approx(gross)
        assert y1.yearly_etf_withdrawal_net == pytest.approx(expected_net)

    def test_negative_leftover_reduces_basis_proportionally(self):
        """Spending exceeds income: ETF is sold monthly; the cost basis falls
        by exactly the sold fraction of the portfolio (average-cost method)."""
        s = Scenario(
            years=5,
            initial_capital=100_000,
            monthly_net_income=0,
            monthly_spending=1_000,
            etf_yearly_return_rate=6 / 100,
            etf_tax_rate=25 / 100,
        )
        df = simulate(s).set_index("year")
        # independent step-by-step reference of the documented rule
        g = 1.06 ** (1 / 12) - 1
        capital, basis = 100_000.0, 100_000.0
        for year in range(1, 6):
            for _ in range(12):
                capital_before_sale = capital * (1 + g)
                capital = capital_before_sale - 1_000
                basis *= 1.0 - min(1_000 / capital_before_sale, 1.0)
            assert df.loc[year, "etf_capital"] == pytest.approx(capital, rel=1e-12)
            assert df.loc[year, "etf_cost_basis"] == pytest.approx(basis, rel=1e-12)
        assert (df.etf_cost_basis >= 0).all()
        assert (df.etf_cost_basis <= df.etf_capital + 1e-6).all()

    def test_withdrawal_updates_basis_across_years(self):
        """The yearly withdrawal must shrink the cost basis proportionally,
        which raises the taxable gain fraction of later withdrawals."""
        s = Scenario(
            years=2,
            initial_capital=100_000,
            etf_yearly_return_rate=6 / 100,
            etf_yearly_withdrawal_rate=4 / 100,
            etf_tax_rate=25 / 100,
        )
        df = simulate(s).set_index("year")
        # year 1: capital 106_000, basis 100_000 before the withdrawal
        assert df.loc[1, "etf_cost_basis"] == pytest.approx(100_000 * 0.96)
        assert df.loc[1, "etf_capital"] == pytest.approx(106_000 * 0.96)
        # year 2 starts from those values
        cap2 = 106_000 * 0.96 * 1.06
        basis2 = 100_000 * 0.96 * 0.96
        assert df.loc[2, "etf_cost_basis"] == pytest.approx(basis2)
        gross2 = cap2 * 0.04
        gain_fraction2 = 1 - (100_000 * 0.96) / cap2
        assert df.loc[2, "yearly_etf_withdrawal_net"] == pytest.approx(
            gross2 * (1 - gain_fraction2 * 0.25)
        )


class TestScenarioComposition:
    def test_rental_income_equivalent_to_extra_income(self):
        """A buy-to-let scenario equals the same scenario with the rental
        income folded into net income (when growth rates match)."""
        common = dict(
            years=15,
            initial_capital=100_000,
            apartment_price=300_000,
            yearly_apartment_price_increase_rate=2 / 100,
            mortgage_interest_rate=3.5 / 100,
            mortgage_down_payment_rate=20 / 100,
            mortgage_fees_rate=10 / 100,
            mortgage_yearly_repayment_rate=2 / 100,
            etf_yearly_return_rate=6 / 100,
        )
        a = simulate(
            Scenario(monthly_net_income=2_000, monthly_rental_income=1_000, **common)
        )
        b = simulate(Scenario(monthly_net_income=3_000, **common))
        pd.testing.assert_series_equal(a.etf_capital, b.etf_capital)
        pd.testing.assert_series_equal(a.total_loan, b.total_loan)

    def test_rent_equivalent_to_extra_spending_for_etf(self):
        """Own rent with 0 growth affects the ETF exactly like spending with
        0 inflation."""
        common = dict(years=10, initial_capital=50_000, monthly_net_income=4_000,
                      etf_yearly_return_rate=6 / 100)
        a = simulate(Scenario(monthly_rent=1_500, **common))
        b = simulate(Scenario(monthly_spending=1_500, **common))
        pd.testing.assert_series_equal(a.etf_capital, b.etf_capital)

    def test_year0_semantics(self):
        s = Scenario(apartment_price=450_000, **BUY_VS_RENT_COMMON)
        df = simulate(s)
        y0 = df.iloc[0]
        assert y0.year == 0
        assert y0.total_loan == pytest.approx(405_000)
        assert y0.property_value == pytest.approx(450_000)
        assert y0.property_equity == pytest.approx(45_000)
        assert y0.etf_capital == pytest.approx(100_000 - 45_000 - 45_000)
        # fees are sunk immediately
        assert y0.estimated_total_capital == pytest.approx(100_000 - 45_000)

    def test_all_factors_together_smoke(self):
        """One scenario exercising every feature at once."""
        s = Scenario(
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
        df = simulate(s)
        assert len(df) == 31
        assert df.notna().all().all()
        assert (df.total_loan >= 0).all()
        assert (df.etf_cost_basis >= 0).all()
        assert df.estimated_total_capital.iloc[-1] > 0


class TestValidation:
    def test_insufficient_capital(self):
        with pytest.raises(ValueError, match="insufficient"):
            Scenario(
                years=10,
                initial_capital=10_000,
                apartment_price=450_000,
                mortgage_down_payment_rate=10 / 100,
                mortgage_fees_rate=10 / 100,
            )

    def test_years_must_be_positive_int(self):
        with pytest.raises(ValueError):
            Scenario(years=0, initial_capital=1_000)
        with pytest.raises(TypeError):
            Scenario(years=10.5, initial_capital=1_000)

    def test_negative_amounts_rejected(self):
        with pytest.raises(ValueError):
            Scenario(years=10, initial_capital=-1)
        with pytest.raises(ValueError):
            Scenario(years=10, initial_capital=0, monthly_spending=-5)

    def test_rates_bounded(self):
        with pytest.raises(ValueError):
            Scenario(years=10, initial_capital=0, etf_tax_rate=1.0)
        with pytest.raises(ValueError):
            Scenario(years=10, initial_capital=0, etf_yearly_return_rate=-1.0)
        with pytest.raises(ValueError):
            Scenario(years=10, initial_capital=0, mortgage_down_payment_rate=1.5)

    def test_replace_produces_new_scenario(self):
        s = Scenario(years=10, initial_capital=1_000)
        s2 = replace(s, etf_yearly_return_rate=5 / 100)
        assert s.etf_yearly_return_rate == 0.0
        assert s2.etf_yearly_return_rate == 5 / 100


class TestHelpers:
    def test_payoff_year(self):
        df = pd.DataFrame({"year": [0, 1, 2, 3], "total_loan": [100, 50, 0, 0]})
        assert payoff_year(df) == 2
        df2 = pd.DataFrame({"year": [0, 1], "total_loan": [100, 50]})
        assert payoff_year(df2) is None

    def test_early_repayment_penalty(self):
        assert calculate_early_repayment_penalty(100_000, 0.038, 0.03, 5) == (
            pytest.approx(100_000 * 0.008 * 5)
        )
        assert calculate_early_repayment_penalty(100_000, 0.038, 0.03, 0) == 0
        # market rate above loan rate -> no damage to the bank -> no penalty
        assert calculate_early_repayment_penalty(100_000, 0.03, 0.04, 5) == 0

    def test_scenario_derived_properties(self):
        s = Scenario(
            years=10,
            initial_capital=100_000,
            apartment_price=450_000,
            mortgage_down_payment_rate=10 / 100,
            mortgage_fees_rate=10 / 100,
            mortgage_interest_rate=3.8 / 100,
            mortgage_yearly_repayment_rate=1.5 / 100,
            sondertilgung_yearly_rate=5 / 100,
        )
        assert s.down_payment == pytest.approx(45_000)
        assert s.purchase_fees == pytest.approx(45_000)
        assert s.initial_loan == pytest.approx(405_000)
        assert s.initial_monthly_mortgage_payment == pytest.approx(
            405_000 * 0.053 / 12
        )
        assert s.max_yearly_sondertilgung == pytest.approx(405_000 * 0.05)
