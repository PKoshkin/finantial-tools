"""Invariants of the engine: cash conservation, closed forms, tax logic.

These do not compare against anything pinned — they check the model against
arithmetic that can be written down independently.
"""

import pandas as pd
import pytest
import scenarios
from src.model import (
    Compounding,
    Etf,
    FinancialModel,
    Home,
    InvestmentProperty,
    Mortgage,
    Rent,
    Salary,
    Spending,
    calculate_early_repayment_penalty,
    payoff_year,
)


def model(components, **kwargs) -> FinancialModel:
    kwargs.setdefault("years", 10)
    return FinancialModel(components=components, **kwargs)


class TestCashConservation:
    def test_no_money_leaks_with_zero_return(self):
        """With a 0% return every euro of leftover either repays the loan
        (annuity or Sondertilgung) or sits in the portfolio. Nothing vanishes."""
        base = scenarios.sondertilgung(5 / 100)
        df = base.replaced("etf", yearly_return_rate=0.0).run()

        home = base.component("home")
        initial_etf = 100_000 - home.down_payment - home.purchase_fees
        for _, row in df[df.year > 0].iterrows():
            upto = df[(df.year > 0) & (df.year <= row.year)]
            expected = (
                initial_etf
                + upto.monthly_leftover.sum() * 12
                - upto.yearly_sondertilgung.sum()
            )
            assert row.etf_capital == pytest.approx(expected, abs=1e-6)

    def test_leftover_accounting(self):
        """invested_capital = initial capital + cumulated leftovers."""
        df = scenarios.buy_vs_rent__buy().run()
        cumulative = 100_000.0
        for _, row in df[df.year > 0].iterrows():
            cumulative += row.monthly_leftover * 12
            assert row.invested_capital == pytest.approx(cumulative, rel=1e-9)


class TestMortgageMath:
    def test_amortization_matches_closed_form(self):
        """Annuity with monthly interest i/12 and fixed payment P:
        L_n = L0*(1+i/12)^n - P*((1+i/12)^n - 1)/(i/12)."""
        mortgage = Mortgage(
            interest_rate=3.8 / 100,
            down_payment_rate=10 / 100,
            fees_rate=10 / 100,
            yearly_repayment_rate=1.5 / 100,
        )
        home = Home(price=450_000, mortgage=mortgage)
        df = model(
            [Salary(monthly_amount=10_000), home, Etf(initial_capital=200_000)], years=5
        ).run()

        loan0 = home.initial_loan
        payment = home.initial_monthly_payment
        rate = mortgage.interest_rate / 12.0
        for year in range(1, 6):
            n = 12 * year
            expected = loan0 * (1 + rate) ** n - payment * ((1 + rate) ** n - 1) / rate
            assert df.set_index("year").loc[year, "total_loan"] == pytest.approx(
                expected, rel=1e-12
            )

    def test_payment_stops_after_payoff_and_cash_flows_to_the_sink(self):
        df = model(
            [
                Salary(monthly_amount=5_000),
                Home(
                    price=200_000,
                    mortgage=Mortgage(
                        interest_rate=3 / 100,
                        down_payment_rate=30 / 100,
                        fees_rate=5 / 100,
                        yearly_repayment_rate=8 / 100,  # aggressive: payoff ~10y
                    ),
                ),
                Etf(initial_capital=100_000),
            ],
            years=12,
        ).run()
        payoff = payoff_year(df)
        assert payoff is not None and payoff < 12

        indexed = df.set_index("year")
        after = indexed.loc[payoff + 1]
        assert after.monthly_interest_payment == 0
        assert after.monthly_loan_repayment == 0
        assert after.monthly_mortgage_payment == 0
        assert after.monthly_leftover == pytest.approx(
            after.monthly_income - after.monthly_spending, rel=1e-9
        )
        assert indexed.loc[payoff + 1 :, "cumulative_interest_paid"].nunique() == 1

    def test_final_payment_capped_at_outstanding(self):
        df = model(
            [
                Salary(monthly_amount=5_000),
                Home(
                    price=200_000,
                    mortgage=Mortgage(
                        interest_rate=3 / 100,
                        down_payment_rate=30 / 100,
                        fees_rate=5 / 100,
                        yearly_repayment_rate=8 / 100,
                    ),
                ),
                Etf(initial_capital=100_000),
            ],
            years=12,
        ).run()
        assert (df.total_loan >= 0).all()

    def test_zero_interest_amortizes_linearly(self):
        df = model(
            [
                Salary(monthly_amount=5_000),
                Home(
                    price=300_000,
                    mortgage=Mortgage(
                        down_payment_rate=20 / 100,
                        fees_rate=10 / 100,
                        yearly_repayment_rate=5 / 100,
                    ),
                ),
                Etf(initial_capital=100_000),
            ],
        ).run().set_index("year")
        loan0 = 240_000
        for year in range(1, 11):
            assert df.loc[year, "total_loan"] == pytest.approx(
                loan0 * (1 - 0.05 * year), abs=1e-6
            )
        assert df.cumulative_interest_paid.iloc[-1] == 0

    def test_full_cash_purchase_has_no_loan(self):
        """down_payment_rate = 1.0: no loan, no payments; price plus fees leave
        the sink and the equity is the full property value."""
        df = model(
            [
                Salary(monthly_amount=3_000),
                Home(
                    price=300_000,
                    yearly_price_increase_rate=2 / 100,
                    mortgage=Mortgage(down_payment_rate=1.0, fees_rate=10 / 100),
                ),
                Etf(initial_capital=400_000, yearly_return_rate=5 / 100),
            ],
            years=5,
        ).run().set_index("year")
        assert (df.total_loan == 0).all()
        assert (df.monthly_mortgage_payment == 0).all()
        assert df.loc[0, "etf_capital"] == pytest.approx(400_000 - 330_000)
        for year in range(0, 6):
            assert df.loc[year, "property_equity"] == pytest.approx(300_000 * 1.02**year)

    def test_mortgage_none_means_paying_cash_without_fees(self):
        df = model(
            [Home(price=100_000), Etf(initial_capital=150_000)], years=3
        ).run().set_index("year")
        assert (df.total_loan == 0).all()
        assert df.loc[0, "etf_capital"] == pytest.approx(50_000)
        assert df.loc[0, "property_equity"] == pytest.approx(100_000)

    def test_refinancing_recomputes_the_annuity(self):
        """Refinancing lowers the payment (smaller base), so the loan is repaid
        more slowly than without it."""

        def build(refinancing_years):
            return model(
                [
                    Salary(monthly_amount=10_000),
                    Home(
                        price=450_000,
                        mortgage=Mortgage(
                            interest_rate=3.8 / 100,
                            down_payment_rate=10 / 100,
                            fees_rate=10 / 100,
                            yearly_repayment_rate=2.5 / 100,
                            refinancing_years=refinancing_years,
                        ),
                    ),
                    Etf(initial_capital=200_000),
                ],
                years=25,
            ).run().set_index("year")

        no_refi, refi = build(0), build(10)
        for year in range(1, 11):  # identical until the first refinancing
            assert no_refi.loc[year, "total_loan"] == pytest.approx(
                refi.loc[year, "total_loan"]
            )
        for year in range(11, 26):  # afterwards the refinanced loan lags behind
            assert refi.loc[year, "total_loan"] >= no_refi.loc[year, "total_loan"] - 1e-6

    def test_sondertilgung_is_capped_and_reduces_interest(self):
        st0 = scenarios.sondertilgung(0).run()
        st5 = scenarios.sondertilgung(5 / 100).run()
        max_st = 450_000 * 0.9 * 0.05
        assert (st5.yearly_sondertilgung <= max_st + 1e-6).all()
        assert st5.cumulative_interest_paid.iloc[-1] < st0.cumulative_interest_paid.iloc[-1]
        payoff0, payoff5 = payoff_year(st0), payoff_year(st5)
        assert payoff5 is not None and (payoff0 is None or payoff5 < payoff0)


class TestEtfMath:
    def test_closed_form_compounding(self):
        """FV = P0*(1+g)^n + c*((1+g)^n - 1)/g with a monthly rate g."""
        df = model(
            [
                Salary(monthly_amount=1_000),
                Etf(initial_capital=50_000, yearly_return_rate=6 / 100),
            ],
        ).run()
        g = 1.06 ** (1 / 12) - 1
        n = 120
        expected = 50_000 * (1 + g) ** n + 1_000 * ((1 + g) ** n - 1) / g
        assert df.etf_capital.iloc[-1] == pytest.approx(expected, rel=1e-12)

    def test_tax_applies_to_gains_only(self):
        df = model(
            [
                Salary(monthly_amount=1_000),
                Etf(
                    initial_capital=50_000,
                    yearly_return_rate=6 / 100,
                    tax_rate=26.375 / 100,
                ),
            ],
        ).run().set_index("year")
        for year in range(1, 11):
            capital = df.loc[year, "etf_capital"]
            basis = df.loc[year, "etf_cost_basis"]
            expected = capital - max(0.0, capital - basis) * 26.375 / 100
            assert df.loc[year, "etf_capital_after_tax"] == pytest.approx(expected)
            assert df.loc[year, "etf_capital_after_tax"] <= capital

    def test_no_tax_on_losses(self):
        df = model(
            [
                Salary(monthly_amount=1_000),
                Etf(
                    initial_capital=50_000,
                    yearly_return_rate=-5 / 100,
                    tax_rate=26.375 / 100,
                ),
            ],
            years=5,
        ).run()
        pd.testing.assert_series_equal(
            df.etf_capital_after_tax, df.etf_capital, check_names=False
        )

    def test_withdrawal_shrinks_capital_geometrically(self):
        df = model(
            [Etf(initial_capital=100_000, yearly_withdrawal_rate=4 / 100)], years=3
        ).run().set_index("year")
        for year in range(1, 4):
            assert df.loc[year, "etf_capital"] == pytest.approx(100_000 * 0.96**year)
            assert df.loc[year, "yearly_etf_withdrawal_gross"] == pytest.approx(
                100_000 * 0.96 ** (year - 1) * 0.04
            )

    def test_withdrawal_is_taxed_on_its_gain_part_only(self):
        df = model(
            [
                Etf(
                    initial_capital=100_000,
                    yearly_return_rate=10 / 100,
                    yearly_withdrawal_rate=5 / 100,
                    tax_rate=25 / 100,
                )
            ],
            years=2,
        ).run().set_index("year")
        # capital before the withdrawal: 110_000, basis 100_000
        gross = 110_000 * 0.05
        gain_fraction = 1 - 100_000 / 110_000
        assert df.loc[1, "yearly_etf_withdrawal_gross"] == pytest.approx(gross)
        assert df.loc[1, "yearly_etf_withdrawal_net"] == pytest.approx(
            gross * (1 - gain_fraction * 0.25)
        )

    def test_selling_reduces_the_basis_proportionally(self):
        """Spending exceeds income: the sink is sold monthly and the cost basis
        falls by exactly the sold fraction of the portfolio."""
        df = model(
            [
                Spending(monthly_amount=1_000),
                Etf(
                    initial_capital=100_000,
                    yearly_return_rate=6 / 100,
                    tax_rate=25 / 100,
                ),
            ],
            years=5,
        ).run().set_index("year")

        # independent step-by-step reference of the documented rule
        g = 1.06 ** (1 / 12) - 1
        capital, basis = 100_000.0, 100_000.0
        for year in range(1, 6):
            for _ in range(12):
                before_sale = capital * (1 + g)
                capital = before_sale - 1_000
                basis *= 1.0 - min(1_000 / before_sale, 1.0)
            assert df.loc[year, "etf_capital"] == pytest.approx(capital, rel=1e-12)
            assert df.loc[year, "etf_cost_basis"] == pytest.approx(basis, rel=1e-12)
        assert (df.etf_cost_basis >= 0).all()
        assert (df.etf_cost_basis <= df.etf_capital + 1e-6).all()

    def test_withdrawal_updates_the_basis_across_years(self):
        df = model(
            [
                Etf(
                    initial_capital=100_000,
                    yearly_return_rate=6 / 100,
                    yearly_withdrawal_rate=4 / 100,
                    tax_rate=25 / 100,
                )
            ],
            years=2,
        ).run().set_index("year")
        # year 1: capital 106_000, basis 100_000 before the withdrawal
        assert df.loc[1, "etf_cost_basis"] == pytest.approx(100_000 * 0.96)
        assert df.loc[1, "etf_capital"] == pytest.approx(106_000 * 0.96)
        # year 2 starts from those values
        capital2 = 106_000 * 0.96 * 1.06
        assert df.loc[2, "etf_cost_basis"] == pytest.approx(100_000 * 0.96 * 0.96)
        gain_fraction2 = 1 - (100_000 * 0.96) / capital2
        assert df.loc[2, "yearly_etf_withdrawal_net"] == pytest.approx(
            capital2 * 0.04 * (1 - gain_fraction2 * 0.25)
        )

    def test_fixed_contribution_is_funded_before_the_sink(self):
        """A fixed-amount portfolio takes its share out of the leftover; the
        sink gets the rest, and the reported leftover covers both."""
        df = model(
            [
                Salary(monthly_amount=3_000),
                Etf(initial_capital=0, monthly_investment=500, name="fixed"),
                Etf(initial_capital=0, name="sink"),
            ],
            years=2,
        ).run().set_index("year")
        assert df.loc[1, "monthly_leftover"] == pytest.approx(3_000)
        assert df.loc[1, "fixed.capital"] == pytest.approx(500 * 12)
        assert df.loc[1, "sink.capital"] == pytest.approx(2_500 * 12)
        assert df.loc[1, "etf_capital"] == pytest.approx(3_000 * 12)


class TestCompounding:
    """The two yearly->monthly conventions, and that each is what it claims."""

    def test_effective_reproduces_the_yearly_rate_exactly(self):
        df = model([Etf(initial_capital=100_000, yearly_return_rate=0.07)], years=3).run()
        for year in range(1, 4):
            assert df.etf_capital.iloc[year] == pytest.approx(
                100_000 * 1.07**year, rel=1e-12
            )

    def test_nominal_compounds_above_the_yearly_rate(self):
        df = model(
            [
                Etf(
                    initial_capital=100_000,
                    yearly_return_rate=0.07,
                    compounding=Compounding.NOMINAL,
                )
            ],
            years=3,
        ).run()
        for year in range(1, 4):
            assert df.etf_capital.iloc[year] == pytest.approx(
                100_000 * (1 + 0.07 / 12) ** (12 * year), rel=1e-12
            )
        # 7% nominal is 7.229% effective, so it must be strictly ahead
        assert df.etf_capital.iloc[-1] > 100_000 * 1.07**3

    def test_mortgage_defaults_to_nominal_accrual(self):
        """A Sollzins accrues rate/12 a month, i.e. above its effective year."""
        mortgage = Mortgage(interest_rate=0.038, down_payment_rate=0.10, fees_rate=0.10)
        assert mortgage.compounding is Compounding.NOMINAL
        assert mortgage.monthly_interest(405_000) == pytest.approx(405_000 * 0.038 / 12)

    def test_mortgage_effective_charges_less_interest(self):
        base = scenarios.sondertilgung(0.0)
        nominal = base.run()
        effective = base.replaced(
            "home.mortgage", compounding=Compounding.EFFECTIVE
        ).run()
        # the same headline rate accrues less when it is read as effective
        assert (
            effective.cumulative_interest_paid.iloc[-1]
            < nominal.cumulative_interest_paid.iloc[-1]
        )
        # ...so the annuity is smaller, and the principal falls more slowly
        assert effective.monthly_mortgage_payment.iloc[0] < (
            nominal.monthly_mortgage_payment.iloc[0]
        )
        assert effective.total_loan.iloc[-1] > nominal.total_loan.iloc[-1]

    def test_the_tilgung_rate_means_the_tilgung_under_both_conventions(self):
        """`yearly_repayment_rate` is the share of the loan repaid in year one.

        Splitting the annuity by the interest convention is what keeps this
        true; charging the nominal annuity while accruing effective interest
        would quietly repay principal a zero-Tilgung loan never agreed to.
        """
        for compounding in (Compounding.NOMINAL, Compounding.EFFECTIVE):
            mortgage = Mortgage(
                interest_rate=0.038,
                down_payment_rate=0.20,
                yearly_repayment_rate=0.0,
                compounding=compounding,
            )
            df = model(
                [
                    Salary(monthly_amount=5_000),
                    Home(price=300_000, mortgage=mortgage),
                    Etf(initial_capital=100_000),
                ],
                years=30,
            ).run()
            assert (df.total_loan == 240_000).all(), f"{compounding} amortized a 0% Tilgung"

            with_tilgung = Mortgage(
                interest_rate=0.038,
                down_payment_rate=0.20,
                yearly_repayment_rate=0.02,
                compounding=compounding,
            )
            first_year = model(
                [
                    Salary(monthly_amount=5_000),
                    Home(price=300_000, mortgage=with_tilgung),
                    Etf(initial_capital=100_000),
                ],
                years=1,
            ).run()
            # the first monthly principal is exactly loan * rate / 12
            assert first_year.monthly_loan_repayment.iloc[0] == 0.0
            assert with_tilgung.annuity_payment(240_000) - with_tilgung.monthly_interest(
                240_000
            ) == pytest.approx(240_000 * 0.02 / 12)

    def test_defaults_reproduce_the_pinned_behaviour(self):
        """The defaults are the conventions the golden fixtures were built on."""
        assert Etf().compounding is Compounding.EFFECTIVE
        assert Spending().compounding is Compounding.EFFECTIVE
        assert Mortgage().compounding is Compounding.NOMINAL

    def test_spending_inflation_follows_its_own_convention(self):
        common = dict(years=2, yearly_inflation_rate=0.12)
        effective = model([Spending(monthly_amount=1_000), Etf()], **common).run()
        nominal = model(
            [Spending(monthly_amount=1_000, compounding=Compounding.NOMINAL), Etf()],
            **common,
        ).run()
        # 12 monthly steps of 12%/12 overshoot an effective 12% year
        assert nominal.monthly_spending.iloc[1] > effective.monthly_spending.iloc[1]
        assert effective.monthly_spending.iloc[1] == pytest.approx(
            sum(1_000 * (1.12 ** (1 / 12)) ** m for m in range(12)) / 12, rel=1e-12
        )

    def test_each_member_computes_its_own_formula(self):
        """Pins which branch is which — a swap breaks every golden fixture."""
        assert Compounding.NOMINAL.monthly_rate(0.12) == 0.12 / 12
        assert Compounding.EFFECTIVE.monthly_rate(0.12) == 1.12 ** (1 / 12) - 1
        assert Compounding.NOMINAL.monthly_charge(1_000, 0.12) == 1_000 * 0.12 / 12
        assert Compounding.EFFECTIVE.monthly_charge(1_000, 0.12) == (
            1_000 * (1.12 ** (1 / 12) - 1)
        )
        # nominal always accrues more than effective for a positive rate
        assert Compounding.NOMINAL.monthly_rate(0.12) > Compounding.EFFECTIVE.monthly_rate(0.12)

    def test_nominal_charge_keeps_the_pinned_operand_order(self):
        """`balance * rate / 12` is not `balance * (rate / 12)` in floating point.

        The golden fixtures were computed with the former; this catches a
        refactor that innocently factors out a monthly rate.
        """
        differ = 0
        for i in range(1, 5_000):
            balance = 100_000 + i * 7.31
            rate = 0.001 + i * 1e-5
            if Compounding.NOMINAL.monthly_charge(balance, rate) != balance * (rate / 12.0):
                differ += 1
            assert Compounding.NOMINAL.monthly_charge(balance, rate) == (
                balance * rate / 12.0
            )
        assert differ > 0, "the two groupings must genuinely differ somewhere"


class TestMonthlyInvested:
    """`monthly_leftover` is the surplus; `monthly_invested` is what reached
    the portfolios. The Sondertilgung reserve is the whole difference."""

    def test_equal_without_a_reserve(self):
        df = scenarios.sondertilgung(0.0).run()
        pd.testing.assert_series_equal(
            df.monthly_invested, df.monthly_leftover, check_names=False
        )

    def test_the_gap_is_what_the_principal_took(self):
        """Exact in every year, including the one the loan is repaid in.

        The reserve is withheld monthly, but only the part the loan can absorb
        is lost to investing — the rest is handed back to the sink at year end.
        """
        df = scenarios.sondertilgung(0.05).run()
        gap = (df.monthly_leftover - df.monthly_invested) * 12
        pd.testing.assert_series_equal(
            gap, df.yearly_sondertilgung, check_names=False, rtol=0, atol=1e-9
        )
        # the payoff year is the interesting one: withheld > applied
        payoff = payoff_year(df)
        assert df.yearly_sondertilgung.iloc[payoff] < 450_000 * 0.9 * 0.05
        assert gap.iloc[payoff + 1 :].eq(0).all()

    def test_the_gap_is_the_full_reserve_while_the_loan_can_take_it(self):
        df = scenarios.sondertilgung(0.05).run()
        reserve = 450_000 * 0.9 * 0.05  # % of the original loan, per year
        gap = (df.monthly_leftover - df.monthly_invested) * 12
        payoff = payoff_year(df)
        assert gap.iloc[1:payoff].tolist() == pytest.approx([reserve] * (payoff - 1))

    def test_negative_when_the_reserve_outgrows_the_surplus(self):
        """The headline case: a positive leftover while the portfolio shrinks."""
        df = scenarios.sondertilgung(0.05).run()
        assert df.monthly_leftover.iloc[1] > 0
        assert df.monthly_invested.iloc[1] < 0
        assert df.etf_capital.iloc[2] < df.etf_capital.iloc[1]

    def test_fixed_contributions_still_count_as_invested(self):
        df = model(
            [
                Salary(monthly_amount=3_000),
                Etf(initial_capital=0, monthly_investment=500, name="fixed"),
                Etf(initial_capital=0, name="sink"),
            ],
            years=2,
        ).run()
        # allocating between two portfolios moves no money out of investments
        assert df.monthly_invested.iloc[1] == pytest.approx(3_000)
        assert df.monthly_invested.iloc[1] == pytest.approx(df.monthly_leftover.iloc[1])

    def test_year_zero_has_nothing_withheld_yet(self):
        """No month has been simulated at year 0, so no claim has been made."""
        df = scenarios.sondertilgung(0.05).run()
        assert df.monthly_invested.iloc[0] == df.monthly_leftover.iloc[0]

    def test_matches_the_portfolio_movement_at_a_zero_return(self):
        """With no growth the portfolio moves by exactly what was invested.

        This is the assertion that pins the column: `monthly_invested` alone
        has to account for the whole change in capital, with no correction term
        that could cancel it out.
        """
        df = model(
            [
                Salary(monthly_amount=4_000),
                Spending(monthly_amount=1_000),
                Home(
                    price=200_000,
                    mortgage=Mortgage(
                        interest_rate=0.03,
                        down_payment_rate=0.20,
                        fees_rate=0.10,
                        yearly_repayment_rate=0.02,
                        sondertilgung_yearly_rate=0.05,
                    ),
                ),
                Etf(initial_capital=100_000, yearly_return_rate=0.0),
            ],
            years=20,
        ).run()
        assert payoff_year(df) is not None, "the release year must be covered"
        for year in range(1, len(df)):
            arrived = df.etf_capital.iloc[year] - df.etf_capital.iloc[year - 1]
            assert df.monthly_invested.iloc[year] * 12 == pytest.approx(arrived, abs=1e-6)

    def test_a_withdrawal_is_not_counted_as_investment(self):
        """Money taken out for spending is reported separately, not netted off."""
        df = model(
            [
                Salary(monthly_amount=1_000),
                Etf(initial_capital=100_000, yearly_withdrawal_rate=0.04),
            ],
            years=3,
        ).run()
        assert (df.yearly_etf_withdrawal_gross.iloc[1:] > 0).all()
        # the surplus was fully invested; the withdrawal is a separate event
        assert df.monthly_invested.iloc[1:].tolist() == pytest.approx([1_000] * 3)
        pd.testing.assert_series_equal(
            df.monthly_invested, df.monthly_leftover, check_names=False
        )


class TestComposition:
    def test_rental_income_equals_extra_salary(self):
        common = dict(
            price=300_000,
            yearly_price_increase_rate=2 / 100,
            mortgage=Mortgage(
                interest_rate=3.5 / 100,
                down_payment_rate=20 / 100,
                fees_rate=10 / 100,
                yearly_repayment_rate=2 / 100,
            ),
        )
        with_rent = model(
            [
                Salary(monthly_amount=2_000),
                InvestmentProperty(monthly_rental_income=1_000, **common),
                Etf(initial_capital=100_000, yearly_return_rate=6 / 100),
            ],
            years=15,
        ).run()
        without = model(
            [
                Salary(monthly_amount=3_000),
                InvestmentProperty(**common),
                Etf(initial_capital=100_000, yearly_return_rate=6 / 100),
            ],
            years=15,
        ).run()
        pd.testing.assert_series_equal(with_rent.etf_capital, without.etf_capital)
        pd.testing.assert_series_equal(with_rent.total_loan, without.total_loan)

    def test_rent_equals_extra_spending_for_the_sink(self):
        common = dict(years=10, yearly_inflation_rate=0.0)
        renting = model(
            [
                Salary(monthly_amount=4_000),
                Rent(monthly_amount=1_500),
                Etf(initial_capital=50_000, yearly_return_rate=6 / 100),
            ],
            **common,
        ).run()
        spending = model(
            [
                Salary(monthly_amount=4_000),
                Spending(monthly_amount=1_500),
                Etf(initial_capital=50_000, yearly_return_rate=6 / 100),
            ],
            **common,
        ).run()
        pd.testing.assert_series_equal(renting.etf_capital, spending.etf_capital)

    def test_year_zero_is_the_post_purchase_state(self):
        row = scenarios.buy_vs_rent__buy().run().iloc[0]
        assert row.year == 0
        assert row.total_loan == pytest.approx(405_000)
        assert row.property_value == pytest.approx(450_000)
        assert row.property_equity == pytest.approx(45_000)
        assert row.etf_capital == pytest.approx(100_000 - 45_000 - 45_000)
        # the fees are sunk immediately
        assert row.estimated_total_capital == pytest.approx(100_000 - 45_000)

    def test_all_factors_together(self):
        df = scenarios.all_factors().run()
        assert len(df) == 31
        assert df.notna().all().all()
        assert (df.total_loan >= 0).all()
        assert (df.etf_cost_basis >= 0).all()
        assert df.estimated_total_capital.iloc[-1] > 0


class TestSweeps:
    def test_replaced_leaves_the_original_alone(self):
        base = scenarios.sondertilgung(0)
        changed = base.replaced("home.mortgage", sondertilgung_yearly_rate=5 / 100)
        assert base.component("home").mortgage.sondertilgung_yearly_rate == 0
        assert changed.component("home").mortgage.sondertilgung_yearly_rate == 5 / 100
        assert changed.run().yearly_sondertilgung.sum() > 0
        assert base.run().yearly_sondertilgung.sum() == 0

    def test_replaced_model_parameters(self):
        base = scenarios.sondertilgung(0)
        assert base.replaced(years=10).run().year.max() == 10
        assert base.years == 30

    def test_replaced_rejects_unknown_targets(self):
        base = scenarios.sondertilgung(0)
        with pytest.raises(ValueError, match="no component named"):
            base.replaced("nope", price=1)
        with pytest.raises(ValueError, match="unknown field"):
            base.replaced("home", nope=1)
        with pytest.raises(ValueError, match="unknown model parameter"):
            base.replaced(nope=1)

    def test_sweep_returns_one_forecast_per_value(self):
        base = scenarios.sondertilgung(0)
        result = base.sweep("home.mortgage", "sondertilgung_yearly_rate", [0.0, 0.05])
        assert set(result) == {0.0, 0.05}
        assert result[0.05].cumulative_interest_paid.iloc[-1] < (
            result[0.0].cumulative_interest_paid.iloc[-1]
        )

    def test_run_is_repeatable(self):
        base = scenarios.all_factors()
        first = base.run()
        first["etf_capital"] = 0.0  # the caller must not be able to poison the cache
        pd.testing.assert_frame_equal(base.run(), scenarios.all_factors().run())


class TestHelpers:
    def test_payoff_year(self):
        df = pd.DataFrame({"year": [0, 1, 2, 3], "total_loan": [100, 50, 0, 0]})
        assert payoff_year(df) == 2
        assert payoff_year(pd.DataFrame({"year": [0, 1], "total_loan": [100, 50]})) is None

    def test_early_repayment_penalty(self):
        assert calculate_early_repayment_penalty(100_000, 0.038, 0.03, 5) == pytest.approx(
            100_000 * 0.008 * 5
        )
        assert calculate_early_repayment_penalty(100_000, 0.038, 0.03, 0) == 0
        # market rate above the loan rate -> no damage to the bank -> no penalty
        assert calculate_early_repayment_penalty(100_000, 0.03, 0.04, 5) == 0

    def test_property_derived_amounts(self):
        home = Home(
            price=450_000,
            mortgage=Mortgage(
                interest_rate=3.8 / 100,
                down_payment_rate=10 / 100,
                fees_rate=10 / 100,
                yearly_repayment_rate=1.5 / 100,
                sondertilgung_yearly_rate=5 / 100,
            ),
        )
        assert home.down_payment == pytest.approx(45_000)
        assert home.purchase_fees == pytest.approx(45_000)
        assert home.initial_loan == pytest.approx(405_000)
        assert home.initial_monthly_payment == pytest.approx(405_000 * 0.053 / 12)
        assert home.max_yearly_sondertilgung == pytest.approx(405_000 * 0.05)

    def test_property_derived_amounts_without_a_mortgage(self):
        home = Home(price=450_000)
        assert home.down_payment == 0
        assert home.purchase_fees == 0
        assert home.initial_loan == 0
        assert home.initial_monthly_payment == 0
        assert home.max_yearly_sondertilgung == 0
