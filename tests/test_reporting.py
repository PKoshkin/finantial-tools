"""Reporting: per-component columns, their aggregates, and multi-component models.

These are the cases the flat pre-refactoring model could not express at all —
more than one flat, more than one salary, more than one portfolio.
"""

import pandas as pd
import pytest
import scenarios
from src.model import (
    FLAT_COLUMNS,
    Etf,
    FinancialModel,
    Home,
    InvestmentProperty,
    Mortgage,
    Rent,
    Salary,
    Spending,
)

# Which per-component keys add up into which flat column.
AGGREGATES = {
    "total_loan": ("total_loan",),
    "property_value": ("property_value",),
    "property_equity": ("property_equity",),
    "etf_capital": ("capital",),
    "etf_cost_basis": ("cost_basis",),
    "etf_capital_after_tax": ("capital_after_tax",),
    "monthly_income": ("monthly_amount",),
    "monthly_rental_income": ("monthly_rental_income",),
    "monthly_interest_payment": ("monthly_interest_payment",),
    "monthly_loan_repayment": ("monthly_loan_repayment",),
    "monthly_mortgage_payment": ("monthly_mortgage_payment",),
    "yearly_sondertilgung": ("yearly_sondertilgung",),
    "cumulative_interest_paid": ("cumulative_interest_paid",),
    "yearly_etf_withdrawal_gross": ("yearly_withdrawal_gross",),
    "yearly_etf_withdrawal_net": ("yearly_withdrawal_net",),
}

KIND_OF_COLUMN = {
    "total_loan": "property",
    "property_value": "property",
    "property_equity": "property",
    "monthly_rental_income": "property",
    "monthly_interest_payment": "property",
    "monthly_loan_repayment": "property",
    "monthly_mortgage_payment": "property",
    "yearly_sondertilgung": "property",
    "cumulative_interest_paid": "property",
    "etf_capital": "etf",
    "etf_cost_basis": "etf",
    "etf_capital_after_tax": "etf",
    "yearly_etf_withdrawal_gross": "etf",
    "yearly_etf_withdrawal_net": "etf",
    "monthly_income": "salary",
}


def portfolio_of_everything() -> FinancialModel:
    """Two of each kind that allows two, so the aggregation is non-trivial."""
    return FinancialModel(
        components=[
            Salary(monthly_amount=4_000, yearly_increase_rate=2 / 100),
            Salary(monthly_amount=2_500, yearly_increase_rate=1 / 100, name="side_job"),
            Spending(monthly_amount=2_000),
            Spending(monthly_amount=400, yearly_increase_rate=5 / 100, name="school"),
            Rent(monthly_amount=1_100, yearly_increase_rate=2 / 100),
            InvestmentProperty(
                price=250_000,
                yearly_price_increase_rate=3 / 100,
                monthly_rental_income=900,
                yearly_rental_income_increase_rate=2 / 100,
                mortgage=Mortgage(
                    interest_rate=3.8 / 100,
                    down_payment_rate=20 / 100,
                    fees_rate=10 / 100,
                    yearly_repayment_rate=2 / 100,
                    refinancing_years=10,
                    sondertilgung_yearly_rate=3 / 100,
                ),
                name="berlin",
            ),
            InvestmentProperty(
                price=180_000,
                yearly_price_increase_rate=2 / 100,
                monthly_rental_income=700,
                yearly_rental_income_increase_rate=1.5 / 100,
                mortgage=Mortgage(
                    interest_rate=4.0 / 100,
                    down_payment_rate=25 / 100,
                    fees_rate=10 / 100,
                    yearly_repayment_rate=3 / 100,
                    refinancing_years=15,
                ),
                name="leipzig",
            ),
            Etf(
                initial_capital=40_000,
                yearly_return_rate=5 / 100,
                tax_rate=26.375 / 100,
                monthly_investment=300,
                name="pension",
            ),
            Etf(
                initial_capital=200_000,
                yearly_return_rate=7 / 100,
                tax_rate=26.375 / 100,
                name="main",
            ),
        ],
        years=20,
        start_age=35,
        yearly_inflation_rate=2 / 100,
    )


class TestAggregation:
    @pytest.mark.parametrize("column", sorted(AGGREGATES))
    def test_flat_column_is_the_sum_of_its_components(self, column):
        model = portfolio_of_everything()
        df = model.run()
        kind = KIND_OF_COLUMN[column]
        keys = AGGREGATES[column]
        contributors = [
            f"{name}.{key}"
            for name, component in zip(model.names, model.components)
            if component.report_kind == kind
            for key in keys
        ]
        assert contributors, f"no component feeds {column}"
        pd.testing.assert_series_equal(
            df[contributors].sum(axis=1), df[column], check_names=False
        )

    def test_spending_and_rent_aggregate_too(self):
        model = portfolio_of_everything()
        df = model.run()
        pd.testing.assert_series_equal(
            df[["spending.monthly_amount", "school.monthly_amount"]].sum(axis=1),
            df["monthly_spending"],
            check_names=False,
        )
        pd.testing.assert_series_equal(
            df["rent.monthly_amount"], df["monthly_rent"], check_names=False
        )

    def test_derived_columns_follow_the_aggregates(self):
        df = portfolio_of_everything().run()
        pd.testing.assert_series_equal(
            df.property_equity + df.etf_capital_after_tax,
            df.estimated_total_capital,
            check_names=False,
        )
        pd.testing.assert_series_equal(
            df.monthly_rent + df.monthly_mortgage_payment,
            df.monthly_housing_cost,
            check_names=False,
        )

    def test_two_salaries_equal_one_of_their_sum(self):
        def build(salaries):
            return FinancialModel(
                components=[*salaries, Etf(initial_capital=10_000, yearly_return_rate=0.05)],
                years=8,
            ).run()

        split = build([Salary(monthly_amount=1_800), Salary(monthly_amount=1_200, name="b")])
        single = build([Salary(monthly_amount=3_000)])
        pd.testing.assert_series_equal(split.etf_capital, single.etf_capital)
        pd.testing.assert_series_equal(split.monthly_income, single.monthly_income)

    def test_two_identical_components_are_not_collapsed(self):
        """Components are frozen dataclasses, so two identical ones compare
        equal — the engine must still keep their state apart."""
        twice = FinancialModel(
            components=[
                Salary(monthly_amount=1_500),
                Salary(monthly_amount=1_500),
                Etf(initial_capital=0),
            ],
            years=5,
        ).run()
        once = FinancialModel(
            components=[Salary(monthly_amount=3_000), Etf(initial_capital=0)],
            years=5,
        ).run()
        assert twice.monthly_income.iloc[1] == pytest.approx(3_000)
        pd.testing.assert_series_equal(twice.etf_capital, once.etf_capital)
        assert twice["salary1.monthly_amount"].iloc[1] == pytest.approx(1_500)
        assert twice["salary2.monthly_amount"].iloc[1] == pytest.approx(1_500)

    def test_two_flats_are_tracked_separately(self):
        model = portfolio_of_everything()
        df = model.run()
        assert df["berlin.total_loan"].iloc[0] == pytest.approx(250_000 * 0.8)
        assert df["leipzig.total_loan"].iloc[0] == pytest.approx(180_000 * 0.75)
        assert df["total_loan"].iloc[0] == pytest.approx(250_000 * 0.8 + 180_000 * 0.75)
        # only berlin has Sondertilgung
        assert df["berlin.yearly_sondertilgung"].sum() > 0
        assert df["leipzig.yearly_sondertilgung"].sum() == 0

    def test_after_tax_is_computed_per_portfolio(self):
        """Two portfolios with different tax rates must not be pooled before
        the tax is applied."""
        model = FinancialModel(
            components=[
                Salary(monthly_amount=1_000),
                Etf(
                    initial_capital=100_000,
                    yearly_return_rate=8 / 100,
                    tax_rate=0.30,
                    monthly_investment=0,
                    name="taxed",
                ),
                Etf(initial_capital=100_000, yearly_return_rate=8 / 100, name="free"),
            ],
            years=5,
        )
        df = model.run()
        assert (df["free.capital_after_tax"] == df["free.capital"]).all()
        assert (df["taxed.capital_after_tax"] < df["taxed.capital"]).iloc[1:].all()
        pd.testing.assert_series_equal(
            df["taxed.capital_after_tax"] + df["free.capital_after_tax"],
            df["etf_capital_after_tax"],
            check_names=False,
        )


class TestColumns:
    def test_layout_is_year_then_flat_then_components(self):
        model = FinancialModel(
            components=[Salary(monthly_amount=1_000), Etf(initial_capital=1_000)],
            years=3,
        )
        columns = list(model.run().columns)
        assert columns[: 1 + len(FLAT_COLUMNS)] == ["year", *FLAT_COLUMNS]
        assert columns[1 + len(FLAT_COLUMNS) :] == [
            "salary.monthly_amount",
            "etf.capital",
            "etf.cost_basis",
            "etf.capital_after_tax",
            "etf.yearly_withdrawal_gross",
            "etf.yearly_withdrawal_net",
        ]

    def test_start_age_adds_an_age_column(self):
        model = FinancialModel([Etf(initial_capital=1)], years=3, start_age=35)
        df = model.run()
        assert list(df.columns[:2]) == ["year", "age"]
        assert df.age.tolist() == [35, 36, 37, 38]
        assert df.age.dtype == df.year.dtype

    def test_without_start_age_there_is_no_age_column(self):
        df = FinancialModel([Etf(initial_capital=1)], years=3).run()
        assert "age" not in df.columns

    def test_age_does_not_disturb_the_forecast(self):
        base = scenarios.all_factors()
        aged = base.replaced(start_age=40)
        pd.testing.assert_frame_equal(
            base.run(), aged.run().drop(columns=["age"]), check_exact=True
        )

    def test_missing_kinds_report_zero(self):
        """A model without property or spending still reports those columns."""
        df = FinancialModel([Etf(initial_capital=1_000)], years=2).run()
        for column in ("total_loan", "property_value", "monthly_spending", "monthly_rent"):
            assert (df[column] == 0).all()

    def test_flat_drops_the_per_component_detail(self):
        from src.model import flat

        df = portfolio_of_everything().run()
        summary = flat(df)
        assert list(summary.columns) == ["year", "age", *FLAT_COLUMNS]
        pd.testing.assert_frame_equal(summary, df[summary.columns])

    def test_home_reports_no_rental_income(self):
        df = FinancialModel(
            [Home(price=100_000), Etf(initial_capital=200_000)], years=2
        ).run()
        assert (df["home.monthly_rental_income"] == 0).all()
        assert (df["monthly_rental_income"] == 0).all()
