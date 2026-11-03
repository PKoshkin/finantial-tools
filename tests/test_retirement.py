"""Retirement: the salary stops, the Regelaltersrente starts, old age is taxed.

Sources (personal documents, kept out of git — `/data/` is ignored; the
figures below are hardcoded from them):

- Both NÜRNBERGER documents in `data/ruerp/` (see tests/test_ruerup.py) end
  with a "Leistungsphase — monatliche Betrachtung": the household's month in
  retirement. The signed contract of 22.10.2024 shows the holder's statutory
  pension, 9,882 EUR from 01.11.2063, next to the spouse's 8,524 EUR (drawn
  since 2059: "zum Alter 71" in 2063), each less 421.76 EUR KV and 207.00 EUR
  PV. That is the KVdR share at 14.6% plus a 1.7% Zusatzbeitrag, and 4.0% PV
  for the childless, both on the 2024 ceiling of 5,175 EUR a month, which the
  illustration does not drift. The offer of 03.09.2024 shows 10,268 EUR for
  the holder alone, less the same 421.76 + 207.00 EUR.
- Both project the pension by a "Näherungsrechnung": the 2024 Rentenwert —
  39.32 EUR in the contract, 37.60 EUR in the offer — raised by 3% a year, to
  124.53 EUR in 2063 and 122.65 EUR in 2064.
- Their income tax is not a tariff: 35% of everything above one
  Grundfreibetrag (11,604 EUR in 2024) per pensioner. It is reproduced below
  to show that, and not compared with the model's assessment.
- The HanseMerkur contract behind the May 2026 payslip (see
  tests/test_real_payslips.py): 638.00 EUR health plus 57.00 EUR care, 349.00
  of the health premium certified Basisabsicherung, a 182.95 EUR
  Pauschalerstattung, and 319.00 + 28.50 EUR employer subsidy on the payslip.

Every engine test runs to 2058 at least: the household only takes a
Rentenbeginn taxed in full. Once a DRV Renteninformation is in `data/ruerp/`,
pin it here: its projected Regelaltersrente is `Retirement.monthly_pension`,
its Entgeltpunkte go through `Retirement.from_entgeltpunkte`.
"""

import pandas as pd
import pytest

from src.model import (
    Earner,
    Etf,
    FinancialModel,
    GermanHousehold,
    PrivateHealthInsurance,
    Retirement,
    RuerupAnnuity,
    RuerupPension,
    TaxSystem,
)
from src.model.german_tax import (
    EarnerFigures,
    SocialContributions,
    assessed_tax,
    besteuerungsanteil,
    income_tax,
    pension_subsidy,
    private_subsidy,
    retiree_contributions,
    social_contributions,
    splitting_tax,
    taxable_income,
)

# The insurer's assumptions for 2024: KV 14.6% + 1.7%, PV 3.4% + 0.6% for the
# childless, the 2024 health ceiling, and no drift of any of it.
INSURER_2024 = TaxSystem(
    kv_extra_rate=0.017, pv_rate=0.034, bbg_health=62_100.0, yearly_wage_growth=0.0
)
FLAT = TaxSystem(yearly_wage_growth=0.0)  # 2026 law, pensions and ceilings held flat
NOBODY = SocialContributions(0.0, 0.0, 0.0, 0.0)
RETIRED_2058 = 12 * (2058 - 2026)  # working months up to a January 2058 Rentenbeginn

HANSEMERKUR = PrivateHealthInsurance(
    monthly_premium=638.0,
    monthly_care_premium=57.0,
    deductible_share=349.0 / 638.0,
    yearly_increase_rate=0.0,
    monthly_rebate=182.95,
)


def retiring(months: int, pension: float, *, gross: float = 0.0, rate=0.0, **changes) -> Earner:
    return Earner(
        monthly_gross=gross,
        retirement=Retirement(working_months=months, monthly_pension=pension, yearly_increase_rate=rate),
        **changes,
    )


def run(household: GermanHousehold, *components, years: int, etf_rate: float = 0.0) -> pd.DataFrame:
    parts = [household, *components, Etf(yearly_return_rate=etf_rate)]
    return FinancialModel(components=parts, years=years).run().set_index("year")


class TestTheDocuments:
    @pytest.mark.parametrize("pension", [9_882.0, 8_524.0, 10_268.0])
    def test_the_kvdr_contributions_of_both_illustrations(self, pension):
        paid = retiree_contributions(pension, INSURER_2024, n_children=0)
        assert paid.health == pytest.approx(421.76, abs=0.005)
        assert paid.care == pytest.approx(207.00, abs=0.005)
        assert paid.pension == paid.unemployment == 0.0

    def test_the_illustrated_month_in_the_engine(self):
        # The signed contract's couple: the spouse retired in 2059, the holder
        # from November 2063. Year 39 is 2064, both of them retired all year.
        couple = GermanHousehold(
            earners=(retiring(454, 9_882.0), retiring(396, 8_524.0)), tax=INSURER_2024
        )
        frame = run(couple, years=39)
        contributions = 421.76 + 207.00
        # "Gesamte Einkünfte vor Steuern" less both contributions: the month's
        # cash before tax, which the illustration nets to 11,383.28 after its
        # 5,765.20 of tax.
        assert frame.loc[39, "household.monthly_amount"] == pytest.approx(
            11_383.28 + 5_765.20, abs=0.01
        )
        assert frame.loc[39, "household.monthly_pension"] == 18_406.0
        assert frame.loc[39, "household.yearly_social_contributions"] == pytest.approx(
            24 * contributions, abs=0.12
        )
        # 2063: twelve months of the spouse's contributions, two of the holder's.
        assert frame.loc[38, "household.yearly_social_contributions"] == pytest.approx(
            14 * contributions, abs=0.07
        )

    def test_the_rentenwert_follows_one_raise_a_year(self):
        # One Entgeltpunkt at the insurer's 3% a year, valued at its 2026 level:
        # model year 38 is 2063, year 39 is 2064.
        for value_2024, year, expected in ((39.32, 38, 124.53), (37.60, 39, 122.65)):
            point = Retirement.from_entgeltpunkte(
                1.0,
                working_months=RETIRED_2058,
                rentenwert=value_2024 * 1.03**2,
                yearly_increase_rate=0.03,
            )
            frame = run(GermanHousehold(earners=(Earner(retirement=point),), joint=False), years=year)
            assert frame.loc[year, "household.monthly_pension"] == pytest.approx(expected, abs=0.005)

    @pytest.mark.parametrize(
        "pensions, grundfreibetraege, tax",
        [
            ((9_882.0, 8_524.0), 2, 5_765.20),
            ((9_882.0, 8_524.0, 3_775.64), 2, 7_086.67),
            ((10_268.0,), 1, 3_255.35),
            ((10_268.0, 4_041.20), 1, 4_669.77),
        ],
    )
    def test_the_illustrated_tax_is_a_flat_rate_not_a_tariff(self, pensions, grundfreibetraege, tax):
        yearly = 12.0 * sum(pensions)
        assert (yearly - grundfreibetraege * 11_604.0) * 0.35 / 12.0 == pytest.approx(tax, abs=0.005)


class TestThePrivateRetiree:
    def test_the_fund_pays_half_the_rates_up_to_half_the_premium(self):
        # 8.75% in 2026: half of 14.6% plus half of the 2.9% Zusatzbeitrag.
        assert pension_subsidy(3_000.0, 638.0, FLAT) == pytest.approx(262.50)
        assert pension_subsidy(4_000.0, 638.0, FLAT) == pytest.approx(319.00)
        # No contribution ceiling: past 5,812.50 a month it keeps rising.
        assert pension_subsidy(7_000.0, 2_000.0, FLAT) == pytest.approx(612.50)

    def test_the_care_premium_loses_its_subsidy(self):
        # Working, the payslip shows 319.00 + 28.50; retired, only health.
        assert private_subsidy(638.0, 57.0, FLAT) == pytest.approx(319.00 + 28.50)
        assert pension_subsidy(10_000.0, 638.0, FLAT) == pytest.approx(319.00)

    def test_a_retired_month_and_its_assessment(self):
        household = GermanHousehold(
            earners=(retiring(RETIRED_2058, 3_000.0, private_health=HANSEMERKUR),), joint=False
        )
        frame = run(household, years=33)
        assert frame.loc[33, "household.monthly_amount"] == pytest.approx(
            3_000.0 - 638.0 - 57.0 + 262.50 + 182.95
        )
        assert frame.loc[33, "household.yearly_social_contributions"] == 0.0
        assert frame.loc[33, "household.yearly_withheld_tax"] == 0.0
        # The certified basis plus care, less the tax-free subsidy, deducted.
        figures = EarnerFigures(
            yearly_gross=0.0,
            social=NOBODY,
            private_deductible=(349.0 + 57.0 - 262.50) * 12.0,
            pension=36_000.0,
        )
        system = FLAT.for_year(32)
        expected = assessed_tax(taxable_income([figures], system, joint=False), system, joint=False, n_children=0)
        assert frame.loc[33, "household.yearly_assessed_tax"] == expected.total
        zve = 36_000.0 - 102.0 - (349.0 + 57.0 - 262.50) * 12.0 - system.special_expenses_allowance
        assert expected.income_tax == income_tax(zve, system)


class TestTheStatutoryRetiree:
    def household(self, *earners) -> GermanHousehold:
        return GermanHousehold(earners=earners, joint=len(earners) == 2, tax=FLAT)

    def test_kvdr_contributions_are_deducted_in_full(self):
        frame = run(self.household(retiring(RETIRED_2058, 3_000.0)), years=33)
        health, care = 3_000.0 * 0.0875, 3_000.0 * 0.042
        assert frame.loc[33, "household.monthly_amount"] == pytest.approx(3_000.0 - health - care)
        assert frame.loc[33, "household.yearly_social_contributions"] == pytest.approx(12 * (health + care))
        # No 4% Krankengeld haircut: a pension pays for no Krankengeld. The
        # 102 EUR allowance is fixed by law; the 36 EUR one drifts here.
        system = FLAT.for_year(32)
        zve = 36_000.0 - 102.0 - 12 * health - 12 * care - system.special_expenses_allowance
        assert frame.loc[33, "household.yearly_assessed_tax"] == income_tax(zve, system)

    def test_the_first_years_tax_is_paid_next_july_then_in_advance(self):
        # Nothing is withheld, and the years before left nothing to pay, so
        # the first retired year owes its whole tax with the next July's
        # settlement. That Bescheid sets the Vorauszahlungen: from then on
        # each year pays the last assessment's tax in advance, and July
        # settles only the change — a small refund here, as the drifting
        # tariff lowers the tax on a flat pension a little every year.
        frame = run(self.household(retiring(RETIRED_2058, 6_000.0)), years=36)
        tax = frame["household.yearly_assessed_tax"]
        prepaid = frame["household.yearly_prepaid_tax"]
        refund = frame["household.yearly_refund_received"]
        assert (frame.loc[33:, "household.yearly_withheld_tax"] == 0.0).all()
        assert prepaid[33] == 0.0
        assert refund[34] == -tax[33]
        for year in (34, 35, 36):
            assert prepaid[year] == pytest.approx(tax[year - 1])
        for year in (35, 36):
            assert refund[year] == pytest.approx(tax[year - 2] - tax[year - 1])
            assert 0.0 < refund[year] < 0.05 * tax[year - 1]

    def test_no_vorauszahlungen_under_400_euros_a_year(self):
        # A smaller pension, whose tax the drifting tariff takes under 400 EUR
        # in the third retired year: the Finanzamt stops asking in advance.
        frame = run(self.household(retiring(RETIRED_2058, 3_000.0)), years=37)
        tax = frame["household.yearly_assessed_tax"]
        prepaid = frame["household.yearly_prepaid_tax"]
        assert tax[34] >= 400.0 > tax[35]
        assert prepaid[35] == pytest.approx(tax[34])
        assert prepaid[36] == prepaid[37] == 0.0
        assert frame.loc[37, "household.yearly_refund_received"] == -tax[36]

    def test_two_pensioners_take_two_allowances(self):
        frame = run(self.household(retiring(RETIRED_2058, 2_000.0), retiring(RETIRED_2058, 1_000.0)), years=33)
        system = FLAT.for_year(32)
        figures = [
            EarnerFigures(
                yearly_gross=0.0,
                social=NOBODY,
                pension=12.0 * pension,
                retiree_health=12.0 * pension * 0.0875,
                retiree_care=12.0 * pension * 0.042,
            )
            for pension in (2_000.0, 1_000.0)
        ]
        zve = taxable_income(figures, system, joint=True)
        assert zve == pytest.approx(
            36_000.0 - 2 * 102.0 - 36_000.0 * (0.0875 + 0.042) - 2 * system.special_expenses_allowance
        )
        assert frame.loc[33, "household.yearly_assessed_tax"] == splitting_tax(zve, system)

    def basisrente(self) -> RuerupPension:
        # Paying 500 EUR a month from the first model month, flat.
        return RuerupPension(
            contribution_months=1,
            elapsed_months=1,
            initial_value=100_000.0,
            annuity=RuerupAnnuity(factor=50.0),
        )

    def test_a_basisrente_owes_no_kvdr_contribution(self):
        alone = run(self.household(retiring(RETIRED_2058, 3_000.0)), years=33)
        both = run(self.household(retiring(RETIRED_2058, 3_000.0)), self.basisrente(), years=33)
        assert both.loc[33, "ruerup.yearly_pension"] == pytest.approx(6_000.0)
        assert (
            both.loc[33, "household.yearly_social_contributions"]
            == alone.loc[33, "household.yearly_social_contributions"]
        )

    def test_the_basisrente_shares_the_first_earners_allowance(self):
        frame = run(self.household(retiring(RETIRED_2058, 3_000.0)), self.basisrente(), years=33)
        health, care = 3_000.0 * 0.0875, 3_000.0 * 0.042
        system = FLAT.for_year(32)
        zve = 36_000.0 + 6_000.0 - 102.0 - 12 * health - 12 * care - system.special_expenses_allowance
        assert frame.loc[33, "household.yearly_assessed_tax"] == income_tax(zve, system)
        # Before the statutory pension starts, the Basisrente has its own.
        assert frame.loc[32, "household.yearly_pension_income"] == pytest.approx(6_000.0)


GROSS = 5_000.0
PENSION = 2_000.0


def single(months: int | None) -> GermanHousehold:
    retirement = None if months is None else Retirement(working_months=months, monthly_pension=PENSION)
    return GermanHousehold(earners=(Earner(monthly_gross=GROSS, retirement=retirement),), joint=False)


@pytest.fixture(scope="module")
def frames():
    """Retiring after April 2058, model year 33, and never retiring."""
    return (
        run(single(RETIRED_2058 + 4), years=35, etf_rate=0.06),
        run(single(None), years=35, etf_rate=0.06),
    )


def pension(year: int) -> float:
    """`PENSION` in today's euros, raised by the default 3.3% wage growth every year."""
    return PENSION * (1.0 + TaxSystem().yearly_wage_growth) ** (year - 1)


class TestTheRetirementYear:
    """A statutory earner who stops working after April 2058, model year 33."""

    def test_four_salary_months_then_eight_pension_months(self, frames):
        retired, working = frames
        assert retired.loc[33, "household.monthly_gross"] == pytest.approx(GROSS * 4 / 12)
        assert retired.loc[33, "household.monthly_pension"] == pytest.approx(pension(33) * 8 / 12)
        assert retired.loc[34, "household.monthly_gross"] == 0.0
        assert retired.loc[34, "household.monthly_pension"] == pytest.approx(pension(34))

    def test_the_salary_comes_first_on_the_calendar(self, frames):
        retired, working = frames
        rate = 1.06 ** (1 / 12) - 1.0
        system = TaxSystem().for_year(32)
        paid = retiree_contributions(pension(33), system, n_children=0)
        months = [working.loc[33, "household.monthly_amount"]] * 4
        months += [pension(33) - paid.health - paid.care] * 8
        months[6] += retired.loc[33, "household.yearly_refund_received"]
        capital = retired.loc[32, "etf_capital"]
        for net in months:
            capital = capital * (1.0 + rate) + net
        assert retired.loc[33, "etf_capital"] == pytest.approx(capital, rel=1e-12)

    def test_withholding_stops_with_the_salary(self, frames):
        retired, working = frames
        assert retired.loc[33, "household.yearly_withheld_tax"] == pytest.approx(
            working.loc[33, "household.yearly_withheld_tax"] * 4 / 12
        )
        assert retired.loc[34, "household.yearly_refund_received"] == (
            retired.loc[33, "household.yearly_withheld_tax"]
            - retired.loc[33, "household.yearly_assessed_tax"]
        )

    def test_one_assessment_covers_both_halves(self, frames):
        retired, _ = frames
        system = TaxSystem().for_year(32)
        # Four months of salary contributions, capped monthly, and the full
        # allowances; eight months of pension and of KVdR contributions.
        social = social_contributions(12 * GROSS, system, n_children=0, private_health=False)
        paid = retiree_contributions(pension(33), system, n_children=0)
        figures = EarnerFigures(
            yearly_gross=4 * GROSS,
            social=SocialContributions(
                *(part / 12 * 4 for part in (social.pension, social.unemployment, social.health, social.care))
            ),
            pension=8 * pension(33),
            retiree_health=8 * paid.health,
            retiree_care=8 * paid.care,
        )
        expected = assessed_tax(taxable_income([figures], system, joint=False), system, joint=False, n_children=0)
        assert retired.loc[33, "household.yearly_assessed_tax"] == expected.total

    def test_a_retirement_beyond_the_horizon_changes_nothing(self):
        later = run(single(12 * 40), years=35, etf_rate=0.06)
        never = run(single(None), years=35, etf_rate=0.06)
        pd.testing.assert_frame_equal(later, never, check_exact=True)

    def test_the_pension_rate_can_be_set(self):
        household = GermanHousehold(
            earners=(retiring(RETIRED_2058, PENSION, rate=0.01),), joint=False
        )
        frame = run(household, years=34)
        assert frame.loc[34, "household.monthly_pension"] == pytest.approx(PENSION * 1.01**33)


class TestValidation:
    def test_a_rentenbeginn_before_2058_is_refused(self):
        with pytest.raises(ValueError, match="2057.*99.5%"):
            GermanHousehold(earners=(retiring(RETIRED_2058 - 1, 1_000.0),), joint=False)
        GermanHousehold(earners=(retiring(RETIRED_2058, 1_000.0),), joint=False)

    def test_the_cohort_table(self):
        for year, share in ((2005, 0.5), (2020, 0.8), (2022, 0.82), (2023, 0.825),
                            (2026, 0.84), (2040, 0.91), (2057, 0.995), (2058, 1.0), (2070, 1.0)):
            assert besteuerungsanteil(year) == pytest.approx(share), year

    def test_the_retirement_is_checked(self):
        with pytest.raises(TypeError, match="working_months"):
            Retirement(working_months=12.0)
        with pytest.raises(ValueError, match="working_months"):
            Retirement(working_months=-1)
        with pytest.raises(ValueError, match="monthly_pension"):
            Retirement(working_months=12, monthly_pension=-1.0)
        with pytest.raises(ValueError, match="yearly_increase_rate"):
            Retirement(working_months=12, yearly_increase_rate=-1.0)
        with pytest.raises(TypeError, match="retirement"):
            Earner(retirement=12)

    def test_entgeltpunkte_are_valued_at_the_start(self):
        point = Retirement.from_entgeltpunkte(79.0, working_months=454)
        assert point.monthly_pension == pytest.approx(79.0 * 40.79)
        assert point.start_year == 2063

    def test_a_long_horizon_needs_the_45_percent_border_to_move(self):
        TaxSystem().for_year(55)
        with pytest.raises(ValueError, match="rich_threshold_frozen=False"):
            TaxSystem().for_year(56)
        TaxSystem(rich_threshold_frozen=False).for_year(80)
