"""The pure German tax functions, pinned to published 2026 worked examples.

Every pinned euro figure below was cross-checked against primary sources
(§32a/§39b EStG as amended for 2026, the Sozialversicherungsrechengrößen-
Verordnung 2026, the GKV-Spitzenverband fact sheet) or against worked
examples derived from them, so these tests are an independent check on the
implementation, not a snapshot of its own output.
"""

import pytest
from src.model.german_tax import (
    EarnerFigures,
    TaxSystem,
    assessed_tax,
    employee_year,
    employer_contributions,
    income_tax,
    private_deductible,
    private_subsidy,
    social_contributions,
    solidarity_surcharge,
    splitting_tax,
    tariff_zones,
    taxable_income,
    withholding,
)

SYSTEM = TaxSystem()


class TestTariff:
    def test_derived_zones_match_the_statutory_2026_coefficients(self):
        zones = tariff_zones(SYSTEM)
        assert zones.progression1 == (914.51, 1400.0)
        assert zones.progression2 == (173.10, 2397.0, 1034.87)
        assert zones.proportional_offset == 11_135.63
        assert zones.rich_offset == 19_470.38

    def test_zone_borders_join_smoothly(self):
        assert income_tax(12_348, SYSTEM) == 0.0
        assert income_tax(17_799, SYSTEM) == 1_034.0
        assert income_tax(17_800, SYSTEM) == 1_035.0
        assert income_tax(69_878, SYSTEM) == 18_213.0
        assert income_tax(69_879, SYSTEM) == 18_213.0

    def test_published_worked_examples(self):
        # BMF example: joint zvE 100,000 -> splitting tax 21,096.
        assert splitting_tax(100_000, SYSTEM) == 21_096.0
        assert income_tax(100_000, SYSTEM) == 30_864.0

    def test_income_is_floored_before_the_tariff(self):
        assert income_tax(100_000.99, SYSTEM) == income_tax(100_000, SYSTEM)

    def test_splitting_tax_is_always_even(self):
        for zve in (30_001, 100_001, 199_999):
            assert splitting_tax(zve, SYSTEM) % 2 == 0

    def test_corner_points_must_be_ordered(self):
        with pytest.raises(ValueError, match="corner points"):
            TaxSystem(basic_allowance=20_000.0)
        with pytest.raises(ValueError, match="corner rates"):
            TaxSystem(entry_rate=0.5)


class TestSolidarity:
    def test_below_the_allowance_nothing_is_due(self):
        assert solidarity_surcharge(20_350, SYSTEM, joint=False) == 0.0
        assert solidarity_surcharge(40_700, SYSTEM, joint=True) == 0.0
        assert solidarity_surcharge(21_096, SYSTEM, joint=True) == 0.0

    def test_the_phase_in_caps_the_surcharge(self):
        # Single, ESt 30,864 (zvE 100,000): the 11.9% cap binds.
        assert solidarity_surcharge(30_864, SYSTEM, joint=False) == 1_251.16

    def test_far_above_the_phase_in_the_full_rate_applies(self):
        assert solidarity_surcharge(80_000, SYSTEM, joint=False) == pytest.approx(4_400.0)


class TestSocialContributions:
    def test_all_bases_cap_at_200k_gross(self):
        # GKV-Spitzenverband fact sheet example, childless.
        social = social_contributions(200_000, SYSTEM, n_children=0, private_health=False)
        assert social.pension == pytest.approx(9_430.20)
        assert social.unemployment == pytest.approx(1_318.20)
        assert social.health == pytest.approx(6_103.13, abs=0.01)
        assert social.care == pytest.approx(1_674.00)
        assert social.total == pytest.approx(18_525.53, abs=0.01)

    def test_children_move_only_the_care_rate(self):
        base = min(200_000, SYSTEM.bbg_health)
        one = social_contributions(200_000, SYSTEM, n_children=1, private_health=False)
        three = social_contributions(200_000, SYSTEM, n_children=3, private_health=False)
        assert one.care == pytest.approx(base * 0.018)
        assert three.care == pytest.approx(base * 0.013)
        assert one.pension == three.pension

    def test_private_health_zeroes_the_health_slots_only(self):
        social = social_contributions(150_000, SYSTEM, n_children=0, private_health=True)
        assert social.health == 0.0
        assert social.care == 0.0
        assert social.pension == pytest.approx(9_430.20)


class TestTaxableIncome:
    def test_the_120k_single_chain(self):
        # 120,000 - 1,230 - 9,430.20 - 5,859.00 - 1,674.00 - 36 = 101,770.80
        figures = EarnerFigures(
            120_000,
            social_contributions(120_000, SYSTEM, n_children=0, private_health=False),
        )
        assert taxable_income([figures], SYSTEM, joint=False) == pytest.approx(101_770.80)

    def test_an_earner_without_wages_gets_no_allowance(self):
        idle = EarnerFigures(
            0.0, social_contributions(0.0, SYSTEM, n_children=0, private_health=False)
        )
        working = EarnerFigures(
            120_000,
            social_contributions(120_000, SYSTEM, n_children=0, private_health=False),
        )
        both = taxable_income([working, idle], SYSTEM, joint=True)
        alone = taxable_income([working], SYSTEM, joint=True)
        assert both == pytest.approx(alone)


class TestAssessment:
    def couple_zve(self, n_children):
        earner = EarnerFigures(
            200_000,
            social_contributions(200_000, SYSTEM, n_children=n_children, private_health=False),
        )
        idle = EarnerFigures(
            0.0, social_contributions(0.0, SYSTEM, n_children=n_children, private_health=False)
        )
        return taxable_income([earner, idle], SYSTEM, joint=True)

    def test_the_200k_single_earner_couple(self):
        zve = self.couple_zve(n_children=1)
        assert zve == pytest.approx(182_153.30)
        assessed = assessed_tax(zve, SYSTEM, joint=True, n_children=0)
        assert assessed.income_tax == 54_232.0
        assert assessed.solidarity == pytest.approx(1_610.30)

    def test_children_swap_the_kindergeld_for_the_allowance_when_it_wins(self):
        zve = self.couple_zve(n_children=1)
        assessed = assessed_tax(zve, SYSTEM, joint=True, n_children=1)
        # Relief 2 * (27,116 - 25,067) = 4,098 beats the 3,108 Kindergeld;
        # the assessed tax is the reduced tariff plus the benefit added back.
        assert assessed.income_tax == 50_134.0 + 3_108.0
        # The Soli base always uses the child-reduced tax (§51a).
        assert assessed.solidarity == pytest.approx(1_122.64)

    def test_at_low_income_the_kindergeld_wins_and_nothing_changes(self):
        with_child = assessed_tax(30_000, SYSTEM, joint=True, n_children=1)
        without = assessed_tax(30_000, SYSTEM, joint=True, n_children=0)
        assert with_child.income_tax == without.income_tax == 870.0


class TestWithholding:
    def test_class_three_matches_the_assessment_for_a_single_earner(self):
        held = withholding(200_000, SYSTEM, tax_class="III", n_children=1)
        assert held.income_tax == 54_232.0

    def test_class_four_withholds_like_a_single_person(self):
        held = withholding(200_000, SYSTEM, tax_class="IV", n_children=1)
        assert held.income_tax == 65_369.0

    def test_class_five_hits_the_top_rate_cap(self):
        # Base 30,674 lies between the first two tiers; the 42% cap binds:
        # floor(0.42 * (30,674 - 14,071)) = 6,973.
        held = withholding(40_000, SYSTEM, tax_class="V", n_children=0)
        assert held.income_tax == 6_973.0

    def test_no_wage_no_withholding(self):
        held = withholding(0.0, SYSTEM, tax_class="V", n_children=1)
        assert held.income_tax == 0.0
        assert held.solidarity == 0.0

    def test_children_touch_the_wage_tax_only_through_the_care_rate(self):
        with_child = withholding(200_000, SYSTEM, tax_class="III", n_children=1)
        without = withholding(200_000, SYSTEM, tax_class="III", n_children=0)
        # The childless care surcharge inflates the Vorsorgepauschale, so the
        # childless wage tax is slightly *lower*; the child allowance itself
        # only ever reduces the Soli base.
        assert without.income_tax < with_child.income_tax
        assert with_child.solidarity < without.solidarity

    def test_unknown_class_is_refused(self):
        with pytest.raises(ValueError, match="tax class"):
            withholding(50_000, SYSTEM, tax_class="II", n_children=0)


class TestPrivateInsurance:
    def test_the_900_euro_premium_example(self):
        # 840 health + 60 care: half of each is under the 2026 caps
        # (508.59 / 104.63 a month), so the subsidy is exactly half.
        assert private_subsidy(840, 60, SYSTEM) == pytest.approx(450.0)
        # Deductible: 0.9 * 840 + 60 - 450 = 366 a month.
        assert private_deductible(840, 60, 0.9, SYSTEM) == pytest.approx(4_392.0)

    def test_a_large_premium_hits_the_caps(self):
        subsidy = private_subsidy(1_200, 250, SYSTEM)
        health_cap = 0.5 * 0.175 * SYSTEM.bbg_health / 12.0
        care_cap = 0.018 * SYSTEM.bbg_health / 12.0
        assert subsidy == pytest.approx(health_cap + care_cap)
        assert subsidy == pytest.approx(508.59 + 104.63, abs=0.01)


class TestInertia:
    def test_the_base_year_is_returned_untouched(self):
        assert SYSTEM.for_year(0) is SYSTEM

    def test_thresholds_and_ceilings_drift_at_their_own_rates(self):
        drifted = SYSTEM.for_year(4)
        assert drifted.basic_allowance == pytest.approx(12_348 * 1.025**4)
        assert drifted.zone3_end == pytest.approx(69_878 * 1.025**4)
        assert drifted.bbg_pension == pytest.approx(101_400 * 1.033**4)
        assert drifted.bbg_health == pytest.approx(69_750 * 1.033**4)
        assert drifted.child_benefit_monthly == pytest.approx(259 * 1.025**4)

    def test_rates_never_drift(self):
        drifted = SYSTEM.for_year(10)
        assert drifted.rv_rate == SYSTEM.rv_rate
        assert drifted.kv_extra_rate == SYSTEM.kv_extra_rate
        assert drifted.top_rate == SYSTEM.top_rate

    def test_the_rich_threshold_stays_frozen_unless_told_otherwise(self):
        assert SYSTEM.for_year(10).rich_threshold == 277_826.0
        thawed = TaxSystem(rich_threshold_frozen=False).for_year(10)
        assert thawed.rich_threshold == pytest.approx(277_826 * 1.025**10)

    def test_a_drifted_tariff_still_taxes_sensibly(self):
        drifted = SYSTEM.for_year(10)
        # More generous borders: the same income owes less tax than today.
        assert income_tax(100_000, drifted) < income_tax(100_000, SYSTEM)
        assert income_tax(drifted.basic_allowance, drifted) == 0.0


class TestOtherVorsorgeCap:
    """The 1,900 EUR bucket (§10 Abs. 4, mirrored by the 2026 payroll rules).

    The unemployment contribution is deductible only as far as it fits the
    bucket next to the health-and-care deduction. A statutory full-time wage
    fills the bucket on its own; a small private Basisabsicherung leaves
    room. The low-basis figures are pinned to the cent against a real DATEV
    payslip (May 2026): gross 100,000, private basis 702/yr, Steuerklasse IV
    -> Lohnsteuer 2,135.66-2,135.67 a month, Soli 52.34.
    """

    def test_small_private_basis_lets_unemployment_fill_the_bucket(self):
        # Vorsorgepauschale = 9,300 (RV) + 702 + min(1,300, 1,900 - 702 = 1,198)
        # -> zvE 100,000 - 1,230 - 36 - 11,200 = 87,534 -> tax 25,628.
        held = withholding(100_000, SYSTEM, tax_class="IV", n_children=0, private_deductible=702.0)
        assert held.income_tax == 25_628.0
        assert held.solidarity == pytest.approx(628.08)

    def test_large_private_basis_leaves_no_room(self):
        # Basis 4,392 > 1,900: the bucket is full, nothing changes against the
        # plain pension + basis pauschale.
        base = 100_000 - 1_230 - 36 - 9_300 - 4_392
        held = withholding(100_000, SYSTEM, tax_class="IV", n_children=0, private_deductible=4_392.0)
        assert held.income_tax == income_tax(base, SYSTEM)

    def test_statutory_full_time_wage_is_unaffected(self):
        # Health + care parts alone exceed 1,900 at any full-time wage, so the
        # unemployment component must not change the pauschale.
        base = 40_000 - 1_230 - 36
        base -= SYSTEM.rv_rate / 2.0 * 40_000
        base -= (SYSTEM.kv_reduced_rate + SYSTEM.kv_extra_rate) / 2.0 * 40_000
        base -= (SYSTEM.pv_rate / 2.0 + SYSTEM.childless_surcharge) * 40_000
        held = withholding(40_000, SYSTEM, tax_class="I", n_children=0)
        assert held.income_tax == income_tax(base, SYSTEM)

    def test_assessment_bucket_mirrors_the_payroll_one(self):
        social = social_contributions(100_000, SYSTEM, n_children=0, private_health=True)
        private = EarnerFigures(yearly_gross=100_000, social=social, private_deductible=702.0)
        zve = taxable_income([private], SYSTEM, joint=False)
        # gross - 1,230 (Arbeitnehmer) - 9,300 (RV) - 702 (basis) - 1,198
        # (unemployment, capped by the bucket) - 36 (Sonderausgaben).
        assert zve == pytest.approx(100_000 - 1_230 - 9_300 - 702 - 1_198 - 36)

    def test_assessment_bucket_is_inert_for_statutory_earners(self):
        social = social_contributions(120_000, SYSTEM, n_children=0, private_health=False)
        earner = EarnerFigures(yearly_gross=120_000, social=social)
        # Health x 0.96 + care exceed 1,900, so the unemployment contribution
        # deducts nothing: the pinned figure from TestTaxableIncome holds.
        assert taxable_income([earner], SYSTEM, joint=False) == pytest.approx(101_770.80)


class TestEmployerContributions:
    def test_the_employer_pays_the_same_but_for_the_care_adjustments(self):
        employee = social_contributions(
            50_000, SYSTEM, n_children=0, private_health=False
        )
        employer = employer_contributions(50_000, SYSTEM, private_health=False)
        assert employer.pension == employee.pension
        assert employer.unemployment == employee.unemployment
        assert employer.health == employee.health
        # The childless surcharge is the employee's alone: 1.8% against 2.4%.
        assert employer.care == pytest.approx(50_000 * 0.018)
        assert employee.care == pytest.approx(50_000 * 0.024)

    def test_children_discount_the_employee_only(self):
        childless = employer_contributions(50_000, SYSTEM, private_health=False)
        with_kids = employer_contributions(50_000, SYSTEM, private_health=False)
        assert childless.care == with_kids.care
        assert social_contributions(
            50_000, SYSTEM, n_children=3, private_health=False
        ).care < childless.care

    def test_each_base_caps_on_its_own(self):
        # SVRV 2026: 101,400 for RV/AV, 69,750 for KV/PV.
        employer = employer_contributions(200_000, SYSTEM, private_health=False)
        assert employer.pension == pytest.approx(101_400 * 0.093)
        assert employer.unemployment == pytest.approx(101_400 * 0.013)
        assert employer.health == pytest.approx(69_750 * 0.0875)
        assert employer.care == pytest.approx(69_750 * 0.018)

    def test_private_health_empties_the_health_slots(self):
        employer = employer_contributions(100_000, SYSTEM, private_health=True)
        assert employer.health == 0.0
        assert employer.care == 0.0
        assert employer.pension == pytest.approx(9_300.0)


class TestEmployeeYear:
    def test_the_100k_statutory_single_chain(self):
        year = employee_year(100_000, SYSTEM)
        # 9.3% + 1.3% on the whole gross (under the 101,400 ceiling), then
        # 8.75% + 2.4% on the 69,750 health ceiling.
        assert year.employee_social.total == pytest.approx(
            9_300 + 1_300 + 6_103.125 + 1_674
        )
        # The employer's care share carries no childless surcharge.
        assert year.employer_social.total == pytest.approx(
            9_300 + 1_300 + 6_103.125 + 1_255.50
        )
        assert year.employer_cost == pytest.approx(117_958.625)
        # zvE = gross - 1,230 - 9,300 - 0.96 x 6,103.125 - 1,674 - 36.
        zve = 100_000 - 1_230 - 9_300 - 5_859.0 - 1_674 - 36
        assert year.assessed.income_tax == income_tax(zve, SYSTEM) == 23_262.0
        assert year.assessed.solidarity == pytest.approx(346.52)
        assert year.net == pytest.approx(100_000 - 18_377.125 - 23_608.52)
        assert year.take_home_share == pytest.approx(0.4918195, abs=1e-7)

    def test_the_refund_bridges_the_two_nets(self):
        year = employee_year(100_000, SYSTEM)
        assert year.net - year.net_paid_out == pytest.approx(year.refund)
        # The 2026 pauschale deducts slightly more than the assessment allows,
        # so a plain statutory wage is under-withheld and owes a little back.
        assert year.refund < 0.0

    def test_an_unpaid_year_has_no_share(self):
        year = employee_year(0.0, SYSTEM)
        assert year.employer_cost == 0.0
        assert year.net == 0.0
        assert year.take_home_share == 0.0

    def test_below_the_basic_allowance_only_the_contributions_bite(self):
        year = employee_year(12_000, SYSTEM)
        assert year.assessed.total == 0.0
        # 21.75% employee, 21.15% employer: the share is pure arithmetic.
        assert year.take_home_share == pytest.approx(0.7825 / 1.2115, abs=1e-6)

    def test_above_both_ceilings_a_raise_costs_the_employer_its_face_value(self):
        lower = employee_year(150_000, SYSTEM)
        higher = employee_year(151_000, SYSTEM)
        assert higher.employer_cost - lower.employer_cost == pytest.approx(1_000.0)
        assert higher.employee_social.total == pytest.approx(lower.employee_social.total)

    def test_the_share_dips_around_the_ceilings_and_then_recovers(self):
        share = lambda gross: employee_year(gross, SYSTEM).take_home_share
        # Rising tax rates push it down while the ceilings still bind, ...
        assert share(20_000) > share(60_000) > share(110_000)
        # ... and once contributions are frozen the flat 42% lets it climb.
        assert share(110_000) < share(200_000) < share(400_000)

    def test_the_married_single_earner_keeps_more(self):
        single = employee_year(100_000, SYSTEM)
        married = employee_year(100_000, SYSTEM, tax_class="III", joint=True)
        assert married.employer_cost == pytest.approx(single.employer_cost)
        # Splitting halves the income before the tariff: the same cost, more net.
        assert married.assessed.income_tax == splitting_tax(
            100_000 - 1_230 - 9_300 - 5_859.0 - 1_674 - 72, SYSTEM
        )
        assert married.net > single.net

    def test_the_tax_class_moves_cash_but_not_the_year(self):
        one = employee_year(100_000, SYSTEM, tax_class="I")
        four = employee_year(100_000, SYSTEM, tax_class="IV")
        five = employee_year(100_000, SYSTEM, tax_class="V")
        assert one.assessed.total == four.assessed.total == five.assessed.total
        assert one.net == pytest.approx(four.net) == pytest.approx(five.net)
        assert five.net_paid_out < one.net_paid_out  # class V front-loads the tax

    def test_children_add_the_kindergeld_and_soften_the_care_rate(self):
        childless = employee_year(100_000, SYSTEM)
        parent = employee_year(100_000, SYSTEM, n_children=2)
        assert parent.net - childless.net == pytest.approx(
            2 * 12 * SYSTEM.child_benefit_monthly
            + (childless.employee_social.care - parent.employee_social.care)
            + (childless.assessed.total - parent.assessed.total)
        )

    def test_a_private_contract_moves_the_employer_share_into_the_subsidy(self):
        # HanseMerkur 2026 contract of the pinned payslip: 638 + 57 a month.
        year = employee_year(
            100_000,
            SYSTEM,
            tax_class="IV",
            monthly_premium=638.0,
            monthly_care_premium=57.0,
            deductible_share=349.0 / 638.0,
        )
        assert year.employer_social.health == 0.0
        assert year.employer_social.care == 0.0
        assert year.employer_subsidy == pytest.approx(12 * (319.00 + 28.50))
        assert year.employer_cost == pytest.approx(100_000 + 10_600 + 4_170)
        assert year.private_premium == pytest.approx(12 * 695.0)

    def test_the_rebate_is_cash_only(self):
        contract = dict(
            monthly_premium=638.0, monthly_care_premium=57.0, deductible_share=0.55
        )
        plain = employee_year(100_000, SYSTEM, **contract)
        rebated = employee_year(100_000, SYSTEM, monthly_rebate=182.95, **contract)
        assert rebated.assessed.total == plain.assessed.total
        assert rebated.withheld.total == plain.withheld.total
        assert rebated.net - plain.net == pytest.approx(12 * 182.95)
