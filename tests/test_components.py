"""Validation: what a component accepts, and what a model refuses to be built from."""

import pytest
from src.model import (
    REMAINDER,
    Etf,
    FinancialModel,
    Home,
    InvestmentProperty,
    Mortgage,
    Rent,
    Salary,
    Spending,
)


class TestComponentValidation:
    def test_amounts_must_not_be_negative(self):
        with pytest.raises(ValueError, match="monthly_amount"):
            Salary(monthly_amount=-1)
        with pytest.raises(ValueError, match="monthly_amount"):
            Spending(monthly_amount=-5)
        with pytest.raises(ValueError, match="monthly_amount"):
            Rent(monthly_amount=-5)
        with pytest.raises(ValueError, match="price"):
            Home(price=-1)
        with pytest.raises(ValueError, match="initial_capital"):
            Etf(initial_capital=-1)
        with pytest.raises(ValueError, match="monthly_rental_income"):
            InvestmentProperty(price=1, monthly_rental_income=-1)

    def test_amounts_must_be_numbers(self):
        with pytest.raises(TypeError, match="monthly_amount"):
            Salary(monthly_amount="1000")

    def test_growth_rates_must_exceed_minus_one(self):
        with pytest.raises(ValueError, match="yearly_increase_rate"):
            Salary(yearly_increase_rate=-1.0)
        with pytest.raises(ValueError, match="yearly_return_rate"):
            Etf(yearly_return_rate=-1.0)
        with pytest.raises(ValueError, match="yearly_price_increase_rate"):
            Home(yearly_price_increase_rate=-1.5)

    def test_etf_rates_are_bounded(self):
        with pytest.raises(ValueError, match="tax_rate"):
            Etf(tax_rate=1.0)
        with pytest.raises(ValueError, match="yearly_withdrawal_rate"):
            Etf(yearly_withdrawal_rate=1.5)
        with pytest.raises(ValueError, match="monthly_investment"):
            Etf(monthly_investment=-100)

    def test_mortgage_rates_are_bounded(self):
        with pytest.raises(ValueError, match="down_payment_rate"):
            Mortgage(down_payment_rate=1.5)
        with pytest.raises(ValueError, match="refinancing_years"):
            Mortgage(refinancing_years=-1)
        with pytest.raises(TypeError, match="refinancing_years"):
            Mortgage(refinancing_years=10.5)
        with pytest.raises(ValueError, match="interest_rate"):
            Mortgage(interest_rate=-0.01)

    def test_mortgage_must_be_a_mortgage(self):
        with pytest.raises(TypeError, match="mortgage"):
            Home(price=100_000, mortgage="none")

    def test_spending_may_override_inflation(self):
        assert Spending(monthly_amount=100).yearly_increase_rate is None
        assert Spending(monthly_amount=100, yearly_increase_rate=0.05).yearly_increase_rate == 0.05
        with pytest.raises(ValueError, match="yearly_increase_rate"):
            Spending(monthly_amount=100, yearly_increase_rate=-1.0)


class TestModelValidation:
    def test_years_must_be_a_positive_int(self):
        with pytest.raises(ValueError, match="years"):
            FinancialModel([Etf()], years=0)
        with pytest.raises(TypeError, match="years"):
            FinancialModel([Etf()], years=10.5)

    def test_start_age_must_be_a_non_negative_int_or_none(self):
        assert FinancialModel([Etf()], years=1, start_age=None).start_age is None
        assert FinancialModel([Etf()], years=1, start_age=0).start_age == 0
        with pytest.raises(TypeError, match="start_age"):
            FinancialModel([Etf()], years=1, start_age=35.5)
        with pytest.raises(ValueError, match="start_age"):
            FinancialModel([Etf()], years=1, start_age=-1)

    def test_model_rates_are_validated(self):
        with pytest.raises(ValueError, match="yearly_inflation_rate"):
            FinancialModel([Etf()], years=1, yearly_inflation_rate=-1.0)
        with pytest.raises(ValueError, match="safe_withdrawal_rate"):
            FinancialModel([Etf()], years=1, safe_withdrawal_rate=-0.01)

    def test_components_must_be_components(self):
        with pytest.raises(TypeError, match="FinancialComponent"):
            FinancialModel([Etf(), "salary"], years=1)

    def test_initial_capital_must_cover_the_purchase(self):
        with pytest.raises(ValueError, match="insufficient"):
            FinancialModel(
                [
                    Home(
                        price=450_000,
                        mortgage=Mortgage(down_payment_rate=0.10, fees_rate=0.10),
                    ),
                    Etf(initial_capital=10_000),
                ],
                years=10,
            )

    def test_initial_capital_may_be_spread_over_portfolios(self):
        FinancialModel(
            [
                Home(price=450_000, mortgage=Mortgage(down_payment_rate=0.10, fees_rate=0.10)),
                Etf(initial_capital=50_000, monthly_investment=0, name="pot"),
                Etf(initial_capital=50_000),
            ],
            years=10,
        )


class TestCardinality:
    def test_at_most_one_home(self):
        with pytest.raises(ValueError, match="at most one Home"):
            FinancialModel([Home(price=1), Home(price=2), Etf(initial_capital=10)], years=1)

    def test_at_most_one_rent(self):
        with pytest.raises(ValueError, match="at most one Rent"):
            FinancialModel([Rent(monthly_amount=1), Rent(monthly_amount=2), Etf()], years=1)

    def test_any_number_of_flats_salaries_and_portfolios(self):
        model = FinancialModel(
            [
                Salary(monthly_amount=1_000),
                Salary(monthly_amount=2_000),
                InvestmentProperty(price=100_000),
                InvestmentProperty(price=200_000),
                Etf(initial_capital=400_000, monthly_investment=0),
                Etf(initial_capital=100_000),
            ],
            years=3,
        )
        assert model.names == ("salary1", "salary2", "flat1", "flat2", "etf1", "etf2")


class TestCashSink:
    def test_a_model_needs_a_sink(self):
        with pytest.raises(ValueError, match="exactly one cash sink"):
            FinancialModel([Salary(monthly_amount=1_000)], years=1)
        with pytest.raises(ValueError, match="exactly one cash sink"):
            FinancialModel([Etf(monthly_investment=100)], years=1)

    def test_only_one_sink_is_allowed(self):
        with pytest.raises(ValueError, match="exactly one cash sink is allowed"):
            FinancialModel([Etf(name="a"), Etf(name="b")], years=1)

    def test_the_default_portfolio_is_the_sink(self):
        assert Etf().monthly_investment is REMAINDER
        assert Etf().is_sink
        assert not Etf(monthly_investment=100).is_sink

    def test_cash_is_a_zero_return_portfolio(self):
        cash = Etf.cash(initial_capital=50_000)
        assert cash.yearly_return_rate == 0.0
        assert cash.tax_rate == 0.0
        df = FinancialModel([Salary(monthly_amount=1_000), cash], years=2).run()
        assert df.etf_capital.iloc[-1] == pytest.approx(50_000 + 1_000 * 24)


class TestNaming:
    def test_a_single_component_keeps_the_bare_default_name(self):
        model = FinancialModel(
            [Salary(monthly_amount=1), InvestmentProperty(price=1), Etf(initial_capital=10)],
            years=1,
        )
        assert model.names == ("salary", "flat", "etf")

    def test_explicit_names_win(self):
        model = FinancialModel(
            [
                InvestmentProperty(price=1, name="berlin"),
                InvestmentProperty(price=2, name="leipzig"),
                Etf(initial_capital=10),
            ],
            years=1,
        )
        assert model.names == ("berlin", "leipzig", "etf")
        assert model.component("berlin").price == 1

    def test_duplicate_names_are_rejected(self):
        with pytest.raises(ValueError, match="duplicate component name"):
            FinancialModel(
                [Salary(name="x"), Etf(name="x")], years=1
            )

    def test_names_may_not_collide_with_result_columns(self):
        with pytest.raises(ValueError, match="collides with a result column"):
            FinancialModel([Etf(name="total_loan")], years=1)
        with pytest.raises(ValueError, match="collides with a result column"):
            FinancialModel([Etf(name="year")], years=1)

    def test_names_must_be_identifier_like(self):
        with pytest.raises(ValueError, match="invalid component name"):
            FinancialModel([Etf(name="My Portfolio")], years=1)

    def test_unknown_component_lookup_lists_the_alternatives(self):
        model = FinancialModel([Salary(monthly_amount=1), Etf()], years=1)
        with pytest.raises(ValueError, match="available: salary, etf"):
            model.component("home")
