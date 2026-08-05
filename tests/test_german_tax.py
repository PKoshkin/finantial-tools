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
