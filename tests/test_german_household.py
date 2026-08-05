"""The GermanHousehold component inside the engine: cash timing and reporting."""

import pytest
from src.model import Earner, Etf, FinancialModel, GermanHousehold, PrivateHealthInsurance
from src.model.german_tax import (
    EarnerFigures,
    TaxSystem,
    assessed_tax,
    private_deductible,
    private_subsidy,
    social_contributions,
    taxable_income,
    withholding,
)

SYSTEM = TaxSystem()


def build(household, *, years=2, etf_rate=0.0):
    return FinancialModel(
        components=[household, Etf(yearly_return_rate=etf_rate)], years=years
    )


def single_earner(*, tax_classes="IV/IV", n_children=1):
    return GermanHousehold(
        earners=(Earner(monthly_gross=200_000 / 12),),
        tax_classes=tax_classes,
        n_children=n_children,
    )


class TestRefundTiming:
    def test_the_first_year_has_no_refund(self):
        df = build(single_earner()).run()
        assert df.loc[1, "household.yearly_refund_received"] == 0.0

    def test_the_refund_is_last_years_overwithholding(self):
        df = build(single_earner()).run()
        expected = (
            df.loc[1, "household.yearly_withheld_tax"]
            - df.loc[1, "household.yearly_assessed_tax"]
        )
        assert df.loc[2, "household.yearly_refund_received"] == pytest.approx(expected)

    def test_the_refund_joins_the_july_cash_and_compounds(self):
        rate = 0.06
        df = build(single_earner(), etf_rate=rate).run()
        monthly_rate = (1.0 + rate) ** (1.0 / 12.0) - 1.0
        capital = 0.0
        for year in (1, 2):
            net = df.loc[year, "household.monthly_amount"]
            refund = df.loc[year, "household.yearly_refund_received"]
            for month in range(1, 13):
                contribution = net + (refund if month == 7 else 0.0)
                capital = capital * (1.0 + monthly_rate) + contribution
        assert df.loc[2, "etf_capital"] == pytest.approx(capital)


class TestTaxClasses:
    def test_classes_change_the_timing_but_not_the_assessed_tax(self):
        df_iii = build(single_earner(tax_classes="III/V")).run()
        df_iv = build(single_earner(tax_classes="IV/IV")).run()
        for year in (1, 2):
            assert df_iii.loc[year, "household.yearly_assessed_tax"] == pytest.approx(
                df_iv.loc[year, "household.yearly_assessed_tax"]
            )
        assert (
            df_iv.loc[2, "household.yearly_refund_received"]
            > df_iii.loc[2, "household.yearly_refund_received"]
        )

    def test_a_years_disposable_income_is_class_independent(self):
        # 12 months of net plus the refund that year triggers must agree.
        df_iii = build(single_earner(tax_classes="III/V")).run()
        df_iv = build(single_earner(tax_classes="IV/IV")).run()
        disposable_iii = (
            12.0 * df_iii.loc[1, "household.monthly_amount"]
            + df_iii.loc[2, "household.yearly_refund_received"]
        )
        disposable_iv = (
            12.0 * df_iv.loc[1, "household.monthly_amount"]
            + df_iv.loc[2, "household.yearly_refund_received"]
        )
        assert disposable_iii == pytest.approx(disposable_iv)


class TestJointTaxation:
    def test_two_average_earners_beat_one_double_earner(self):
        one = GermanHousehold(
            earners=(Earner(monthly_gross=200_000 / 12), Earner(monthly_gross=0.0)),
            tax_classes="IV/IV",
        )
        two = GermanHousehold(
            earners=(
                Earner(monthly_gross=100_000 / 12),
                Earner(monthly_gross=100_000 / 12),
            ),
            tax_classes="IV/IV",
        )
        df_one = build(one).run()
        df_two = build(two).run()
        # Same household gross, but two employee allowances and two sets of
        # contribution ceilings push the two-earner zvE (and tax) well below.
        assert (
            df_two.loc[1, "household.yearly_assessed_tax"]
            < df_one.loc[1, "household.yearly_assessed_tax"]
        )

    def test_the_component_agrees_with_the_pure_functions(self):
        df = build(single_earner(n_children=0)).run()
        figures = [
            EarnerFigures(
                200_000,
                social_contributions(200_000, SYSTEM, n_children=0, private_health=False),
            ),
        ]
        assessment = assessed_tax(
            taxable_income(figures, SYSTEM, joint=True), SYSTEM, joint=True, n_children=0
        )
        assert df.loc[1, "household.yearly_assessed_tax"] == pytest.approx(assessment.total)
        held = withholding(200_000, SYSTEM, tax_class="IV", n_children=0)
        assert df.loc[1, "household.yearly_withheld_tax"] == pytest.approx(held.total)


class TestPrivateInsurance:
    def test_going_private_moves_the_net_by_the_premium_arithmetic(self):
        gross = 150_000
        contract = PrivateHealthInsurance(monthly_premium=840, monthly_care_premium=60)
        public = GermanHousehold(earners=(Earner(monthly_gross=gross / 12),))
        private = GermanHousehold(
            earners=(Earner(monthly_gross=gross / 12, private_health=contract),)
        )
        df_public = build(public).run()
        df_private = build(private).run()

        social = social_contributions(gross, SYSTEM, n_children=0, private_health=False)
        deductible = private_deductible(840, 60, contract.deductible_share, SYSTEM)
        held_public = withholding(gross, SYSTEM, tax_class="III", n_children=0)
        held_private = withholding(
            gross, SYSTEM, tax_class="III", n_children=0, private_deductible=deductible
        )
        expected_shift = (
            (social.health + social.care) / 12.0
            - (840 + 60)
            + private_subsidy(840, 60, SYSTEM)
            + (held_public.total - held_private.total) / 12.0
        )
        shift = (
            df_private.loc[1, "household.monthly_amount"]
            - df_public.loc[1, "household.monthly_amount"]
        )
        assert shift == pytest.approx(expected_shift)


class TestReporting:
    def test_the_flat_income_column_is_the_household_net(self):
        df = build(single_earner()).run()
        for year in (0, 1, 2):
            assert df.loc[year, "monthly_income"] == pytest.approx(
                df.loc[year, "household.monthly_amount"]
            )

    def test_the_initial_row_previews_the_first_year(self):
        df = build(single_earner()).run()
        assert df.loc[0, "household.monthly_amount"] == pytest.approx(
            df.loc[1, "household.monthly_amount"]
        )
        assert df.loc[0, "household.yearly_refund_received"] == 0.0

    def test_the_tax_system_is_sweepable(self):
        model = build(single_earner())
        expensive = model.replaced("household.tax", kv_extra_rate=0.045)
        base_net = model.run().loc[1, "household.monthly_amount"]
        expensive_net = expensive.run().loc[1, "household.monthly_amount"]
        assert expensive_net < base_net


class TestValidation:
    def test_the_household_needs_one_or_two_earners(self):
        with pytest.raises(ValueError, match="one or two"):
            GermanHousehold()
        with pytest.raises(ValueError, match="one or two"):
            GermanHousehold(earners=(Earner(), Earner(), Earner()))

    def test_two_earners_must_file_jointly(self):
        with pytest.raises(ValueError, match="jointly"):
            GermanHousehold(earners=(Earner(), Earner()), joint=False)

    def test_only_the_two_class_combinations_exist(self):
        with pytest.raises(ValueError, match="tax_classes"):
            GermanHousehold(earners=(Earner(),), tax_classes="I/II")

    def test_amounts_and_shares_are_checked(self):
        with pytest.raises(ValueError, match="monthly_gross"):
            Earner(monthly_gross=-1)
        with pytest.raises(ValueError, match="deductible_share"):
            PrivateHealthInsurance(monthly_premium=800, deductible_share=1.5)
        with pytest.raises(ValueError, match="n_children"):
            GermanHousehold(earners=(Earner(),), n_children=-1)
        with pytest.raises(TypeError, match="n_children"):
            GermanHousehold(earners=(Earner(),), n_children=1.5)

    def test_the_household_is_a_singleton(self):
        with pytest.raises(ValueError):
            build_two = [
                GermanHousehold(earners=(Earner(monthly_gross=5_000),)),
                GermanHousehold(earners=(Earner(monthly_gross=4_000),), name="second"),
                Etf(),
            ]
            FinancialModel(components=build_two, years=1)
