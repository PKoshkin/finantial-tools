"""The AfA schedule: a special rate for the first years, and the write-off.

The schedule is pinned against a third calculator sheet (kept out of git,
`data/buy_wg/new_calc/Viernheim_Kalkulation_Jahre_1-4.pdf` and
`..._ab_Jahr_5.pdf`): a Viernheim flat at 470,000 EUR, 11.57% purchase
fees, 80% building share, financed 90% at 4.70% with 1.30% Tilgung. The
broker's sheets apply 8.85% for the first four years (a 5% Sonder-AfA on top
of 3.85% from a Restnutzungsdauer-Gutachten) and 3.85% from year five on,
each as a static first-year snapshot at a flat 42% tax rate. The engine runs
them as one schedule, amortizes monthly and taxes at the real tariff, so the
tolerances below follow the ones in tests/test_rental_taxation.py.
"""

from dataclasses import replace

import pytest

from src.model import (
    Earner,
    Etf,
    FinancialModel,
    GermanHousehold,
    InvestmentProperty,
    Mortgage,
    OperatingCosts,
    RentalTaxation,
)

MARGINAL = 0.42 * 1.055  # top-zone rate with full Solidaritaetszuschlag
SCHEDULE = RentalTaxation(
    afa_rate=0.0385, building_share=0.80, special_afa_rate=0.05, special_afa_years=4
)


def household() -> GermanHousehold:
    # 200k gross: flat 42% zone, Soli fully phased in, as in test_rental_taxation.
    return GermanHousehold(earners=(Earner(monthly_gross=200_000 / 12.0),), joint=False)


def viernheim(taxation: RentalTaxation = SCHEDULE) -> InvestmentProperty:
    """The broker's sheet: 2,815 EUR in the rent field, 673 EUR of costs, no reserve."""
    return InvestmentProperty(
        price=470_000,
        yearly_price_increase_rate=0.02,
        monthly_rental_income=2_815,
        yearly_rental_income_increase_rate=0.02,
        mortgage=Mortgage(
            interest_rate=0.047,
            down_payment_rate=0.10,
            fees_rate=0.1157,
            yearly_repayment_rate=0.013,
        ),
        operating_costs=OperatingCosts(monthly_deductible=673.0, yearly_increase_rate=0.02),
        taxation=taxation,
        name="flat",
    )


def run(flat: InvestmentProperty | None, *, years: int = 6):
    components = [household(), Etf(initial_capital=150_000, yearly_return_rate=0.0)]
    if flat is not None:
        components.insert(1, flat)
    return FinancialModel(components=components, years=years).run().set_index("year")


def tax_delta(with_flat, alone, year: int) -> float:
    """Extra tax the flat causes: positive means a refund (a rental loss)."""
    return (
        alone.loc[year, "household.yearly_assessed_tax"]
        - with_flat.loc[year, "household.yearly_assessed_tax"]
    )


@pytest.fixture(scope="module")
def alone():
    return run(None)


@pytest.fixture(scope="module")
def frame():
    return run(viernheim())


class TestTheSheets:
    def test_the_base_and_both_monthly_figures(self):
        flat = viernheim()
        # sheet: Basis fuer langfristige Abschreibung 419,503 = 524,379 x 80%
        assert flat.afa_base == pytest.approx(419_503.2, rel=1e-12)
        assert flat.purchase_fees == pytest.approx(54_379.0, rel=1e-12)
        # sheet "Jahre 1-4": Abschreibung 3,094 a month; "ab Jahr 5": 1,346
        assert flat.monthly_afa == pytest.approx(3_094.0, abs=0.5)
        assert flat.yearly_afa(5, written_off=0.0) / 12.0 == pytest.approx(1_346.0, abs=0.5)
        # The equity the sheet asks for: 10% of the price plus the fees.
        assert flat.down_payment + flat.purchase_fees == pytest.approx(101_379.0, rel=1e-12)
        assert flat.initial_monthly_payment == pytest.approx(2_115.0, rel=1e-12)

    def test_the_schedule_switches_after_year_four(self, frame):
        for year in (0, 1, 2, 3, 4):
            assert frame.loc[year, "flat.monthly_afa"] == pytest.approx(3_094.0, abs=0.5)
        for year in (5, 6):
            assert frame.loc[year, "flat.monthly_afa"] == pytest.approx(1_346.0, abs=0.5)

    def test_year_one_against_the_first_sheet(self, frame, alone):
        # sheet Nebenrechnung x12: -2,609 x 12. Monthly amortization pays a
        # little less interest than the sheet's static 1,657, hence the slack.
        assert frame.loc[1, "flat.yearly_taxable_result"] == pytest.approx(-31_308.0, abs=200.0)
        assert tax_delta(frame, alone, 1) == pytest.approx(31_308.0 * MARGINAL, abs=150.0)
        assert tax_delta(frame, alone, 1) > 0

    def test_the_lower_rate_alone_against_the_second_sheet(self, alone):
        # The "ab Jahr 5" sheet is the same first-year snapshot at 3.85% only:
        # Nebenrechnung -861 x 12.
        lower = run(viernheim(RentalTaxation(afa_rate=0.0385, building_share=0.80)))
        assert lower.loc[1, "flat.monthly_afa"] == pytest.approx(1_346.0, abs=0.5)
        assert lower.loc[1, "flat.yearly_taxable_result"] == pytest.approx(-10_332.0, abs=200.0)
        assert tax_delta(lower, alone, 1) == pytest.approx(10_332.0 * MARGINAL, abs=150.0)

    def test_the_taxable_result_keeps_its_closed_form_across_the_switch(self, frame):
        for year in (1, 4, 5, 6):
            row = frame.loc[year]
            expected = (
                12.0 * row["flat.monthly_rental_income"]
                - 12.0 * 673.0 * 1.02 ** (year - 1)
                - 12.0 * row["flat.monthly_interest_payment"]
                - 12.0 * row["flat.monthly_afa"]
            )
            assert row["flat.yearly_taxable_result"] == pytest.approx(expected, rel=1e-9)

    def test_the_refund_drops_with_the_rate(self, frame, alone):
        # Year 5's smaller deduction shows in year 6's July settlement.
        in_four = frame.loc[5, "household.yearly_refund_received"] - alone.loc[5, "household.yearly_refund_received"]
        in_five = frame.loc[6, "household.yearly_refund_received"] - alone.loc[6, "household.yearly_refund_received"]
        assert in_four > in_five + 0.9 * 12.0 * (3_094.0 - 1_346.0) * MARGINAL


def small(taxation: RentalTaxation) -> InvestmentProperty:
    """A cash-bought building, all of it depreciable: the AfA is the whole result."""
    return InvestmentProperty(price=120_000, monthly_rental_income=0.0, taxation=taxation, name="flat")


def run_small(taxation: RentalTaxation, years: int):
    return FinancialModel(
        components=[household(), small(taxation), Etf(initial_capital=200_000, yearly_return_rate=0.0)],
        years=years,
    ).run().set_index("year")


class TestTheWriteOff:
    def test_a_whole_number_of_years(self):
        frame = run_small(RentalTaxation(afa_rate=0.25, building_share=1.0), years=6)
        for year in (1, 2, 3, 4):
            assert frame.loc[year, "flat.yearly_taxable_result"] == pytest.approx(-30_000.0)
        for year in (5, 6):
            assert frame.loc[year, "flat.monthly_afa"] == 0.0
            assert frame.loc[year, "flat.yearly_taxable_result"] == 0.0

    def test_the_last_year_takes_what_is_left(self):
        frame = run_small(RentalTaxation(afa_rate=0.3, building_share=1.0), years=5)
        deducted = -frame.loc[1:, "flat.yearly_taxable_result"]
        assert list(deducted.round(6)) == pytest.approx([36_000.0, 36_000.0, 36_000.0, 12_000.0, 0.0])
        assert deducted.sum() == pytest.approx(120_000.0)

    def test_the_special_rate_shortens_the_write_off(self):
        frame = run_small(
            RentalTaxation(afa_rate=0.3, building_share=1.0, special_afa_rate=0.2, special_afa_years=2),
            years=4,
        )
        deducted = -frame.loc[1:, "flat.yearly_taxable_result"]
        assert list(deducted.round(6)) == pytest.approx([60_000.0, 60_000.0, 0.0, 0.0])

    def test_nothing_is_deducted_past_the_base_in_a_long_run(self):
        # 3% for 40 years would deduct 120% of the base without the cap.
        taxation = RentalTaxation(afa_rate=0.03, building_share=1.0)
        frame = run_small(taxation, years=40)
        deducted = -frame.loc[1:, "flat.yearly_taxable_result"]
        assert deducted.sum() == pytest.approx(120_000.0)
        assert (deducted.loc[35:] == 0.0).all()
        assert deducted.loc[34] == pytest.approx(0.01 * 120_000.0, abs=1e-6)

    def test_the_household_sees_the_capped_result(self):
        frame = run_small(RentalTaxation(afa_rate=0.5, building_share=1.0), years=3)
        alone = FinancialModel(
            components=[household(), Etf(initial_capital=200_000, yearly_return_rate=0.0)], years=3
        ).run().set_index("year")
        assert frame.loc[3, "household.yearly_rental_result"] == 0.0
        assert frame.loc[3, "household.yearly_assessed_tax"] == alone.loc[3, "household.yearly_assessed_tax"]


class TestDefaults:
    def test_a_plain_taxation_is_unchanged(self):
        # No special rate, horizon inside the write-off: the schedule is a constant.
        plain = RentalTaxation(afa_rate=0.03, building_share=0.96)
        assert plain.special_afa_rate == 0.0
        assert plain.special_afa_years == 0
        for year in (1, 10, 33):
            assert plain.yearly_rate(year) == 0.03
        flat = replace(viernheim(plain), price=510_000, mortgage=Mortgage(fees_rate=0.12))
        assert flat.monthly_afa == pytest.approx(510_000 * 1.12 * 0.96 * 0.03 / 12.0, rel=1e-12)
        assert flat.yearly_afa(20, written_off=0.0) == flat.yearly_afa(1, written_off=0.0)

    def test_an_untaxed_flat_has_no_afa(self):
        flat = InvestmentProperty(price=100_000)
        assert flat.monthly_afa == 0.0
        assert flat.yearly_afa(1, written_off=0.0) == 0.0


class TestValidation:
    def test_the_special_rate_is_checked(self):
        with pytest.raises(ValueError, match="special_afa_rate"):
            RentalTaxation(afa_rate=0.02, building_share=0.8, special_afa_rate=-0.01)
        with pytest.raises(ValueError, match="afa_rate plus special_afa_rate"):
            RentalTaxation(afa_rate=0.6, building_share=0.8, special_afa_rate=0.5)
        RentalTaxation(afa_rate=0.5, building_share=0.8, special_afa_rate=0.5, special_afa_years=1)

    def test_the_special_years_are_checked(self):
        with pytest.raises(TypeError, match="special_afa_years"):
            RentalTaxation(afa_rate=0.02, building_share=0.8, special_afa_years=4.0)
        with pytest.raises(TypeError, match="special_afa_years"):
            RentalTaxation(afa_rate=0.02, building_share=0.8, special_afa_years=True)
        with pytest.raises(ValueError, match="special_afa_years"):
            RentalTaxation(afa_rate=0.02, building_share=0.8, special_afa_years=-1)
