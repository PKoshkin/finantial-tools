"""The Basisrente against two real NÜRNBERGER documents, and inside the engine.

Sources (two PDFs in `data/ruerp/`, kept out of git — `/data/` is ignored;
the figures below are hardcoded from them), both for tariff NFX3208T:

- The signed contract of 27.10.2024: 500 EUR a month from 01.11.2024, 468
  contributions, Rentenbeginn 01.11.2063, five equity funds. Its PIB quotes
  2.5% acquisition cost on the 234,000 EUR contribution sum (5,850 EUR),
  4.5% of every contribution, and a 0.80% capital charge that includes the
  funds' own costs.
- The offer of 03.09.2024 for the same person: 480 contributions from
  01.10.2024 into an S&P 500 ETF, 6,000 EUR acquisition cost, a 0.49%
  capital charge including the fund.

Both tabulate the contract value at every anniversary for a fund growth of
0/3/7/9% (the Todesfallleistung before Rentenbeginn is the contract value).
One cost model reproduces both documents with the PIB's own rates and a
0.40% insurer capital charge, so the tables pin the tariff, not a fit.

The residual comes from the insurer's own clock: the model compounds twelve
equal months, the insurer appears to count calendar days. The model runs a
few euros high from the first year on — 6e-4 of a small early value, down
to 1e-4 of the value after four decades, never more than 20 EUR until the
value passes 133,000 EUR — hence `rel=1.5e-4, abs=20` for the tables (92%
of it used at worst) and `rel=1e-4` for the converted capital. The annuity
formula, applied to the insurer's own capital, matches to the cent.
"""

import pandas as pd
import pytest

from src.model import (
    Earner,
    Etf,
    FinancialModel,
    GermanHousehold,
    RuerupAnnuity,
    RuerupCosts,
    RuerupPension,
    Salary,
)
from src.model.german_tax import (
    EarnerFigures,
    SocialContributions,
    TaxSystem,
    assessed_tax,
    basisrente_cap,
    social_contributions,
    taxable_income,
)

SYSTEM = TaxSystem()

# The 2024 Rechnungsgrundlagen plus the declared teildynamische Bonusrente.
DECLARED_2024 = RuerupAnnuity(factor=24.996271, bonus_rate=0.0187)
# The documents' 0% case assumes a Zinsüberschuss one point lower, which
# leaves a 1.19% Zusatzrente.
DECLARED_LOWER = RuerupAnnuity(factor=24.996271, bonus_rate=0.0119)

SIGNED_VALUES = {
    0.00: [4_549, 9_081, 13_595, 18_091, 22_570, 28_200, 33_807, 39_391, 44_953, 50_493,
           56_011, 61_507, 66_981, 72_433, 77_863, 83_272, 88_659, 94_025, 99_369, 104_691,
           109_993, 115_273, 120_533, 125_771, 130_988, 136_185, 141_360, 146_516, 151_650,
           156_764, 161_858, 166_931, 171_984, 177_017, 182_029, 187_022, 191_995, 196_948,
           201_881],
    0.03: [4_611, 9_342, 14_197, 19_177, 24_288, 30_714, 37_307, 44_070, 51_009, 58_127,
           65_429, 72_921, 80_606, 88_491, 96_579, 104_877, 113_390, 122_123, 131_082,
           140_273, 149_702, 159_375, 169_298, 179_479, 189_923, 200_637, 211_629, 222_905,
           234_474, 246_341, 258_516, 271_006, 283_819, 296_964, 310_450, 324_284, 338_477,
           353_037, 367_974],
    0.07: [4_693, 9_695, 15_027, 20_710, 26_767, 34_426, 42_590, 51_290, 60_562, 70_443,
           80_974, 92_197, 104_157, 116_904, 130_488, 144_966, 160_395, 176_838, 194_362,
           213_037, 232_940, 254_152, 276_757, 300_849, 326_523, 353_886, 383_046, 414_124,
           447_244, 482_541, 520_158, 560_248, 602_972, 648_505, 697_031, 748_746, 803_860,
           862_596, 925_194],
    0.09: [4_733, 9_873, 15_454, 21_513, 28_092, 36_450, 45_523, 55_374, 66_068, 77_678,
           90_283, 103_967, 118_823, 134_952, 152_462, 171_471, 192_109, 214_515, 238_839,
           265_247, 293_916, 325_041, 358_832, 395_517, 435_344, 478_582, 525_523, 576_484,
           631_811, 691_876, 757_085, 827_879, 904_737, 988_178, 1_078_765, 1_177_111,
           1_283_879, 1_399_793, 1_525_634],
}

OFFER_VALUES = {
    0.00: [4_519, 9_021, 13_505, 17_972, 22_422, 28_052, 33_659, 39_244, 44_807, 50_348,
           55_866, 61_363, 66_837, 72_290, 77_721, 83_130, 88_517, 93_883, 99_228, 104_551,
           109_854, 115_134, 120_394, 125_633, 130_851, 136_048, 141_224, 146_380, 151_515,
           156_630, 161_724, 166_797, 171_851, 176_884, 181_898, 186_891, 191_864, 196_818,
           201_751, 206_665],
    0.03: [4_581, 9_281, 14_103, 19_051, 24_127, 30_550, 37_138, 43_897, 50_831, 57_945,
           65_243, 72_729, 80_410, 88_289, 96_373, 104_665, 113_172, 121_900, 130_853,
           140_038, 149_461, 159_128, 169_045, 179_218, 189_656, 200_363, 211_348, 222_617,
           234_178, 246_038, 258_205, 270_687, 283_492, 296_629, 310_105, 323_931, 338_114,
           352_665, 367_592, 382_906],
    0.07: [4_662, 9_631, 14_928, 20_573, 26_590, 34_239, 42_390, 51_076, 60_334, 70_201,
           80_715, 91_921, 103_864, 116_591, 130_155, 144_611, 160_016, 176_434, 193_932,
           212_579, 232_452, 253_631, 276_203, 300_257, 325_893, 353_214, 382_331, 413_361,
           446_431, 481_675, 519_235, 559_264, 601_924, 647_388, 695_840, 747_476, 802_507,
           861_155, 923_657, 990_268],
    0.09: [4_702, 9_808, 15_352, 21_372, 27_907, 36_249, 45_305, 55_137, 65_811, 77_399,
           89_980, 103_638, 118_466, 134_564, 152_041, 171_014, 191_613, 213_976, 238_254,
           264_612, 293_227, 324_293, 358_020, 394_635, 434_386, 477_542, 524_395, 575_260,
           630_481, 690_432, 755_518, 826_178, 902_890, 986_173, 1_076_588, 1_174_747,
           1_281_314, 1_397_007, 1_522_609, 1_658_970],
}

# Capital converted at Rentenbeginn, and the possible monthly pension from it.
SIGNED_START = {0.00: (201_881, 704.83), 0.03: (368_881, 1_496.91),
                0.07: (930_425, 3_775.64), 0.09: (1_536_629, 6_235.58)}
OFFER_START = {0.00: (206_665, 721.54), 0.03: (383_850, 1_557.65),
               0.07: (995_867, 4_041.20), 0.09: (1_670_926, 6_780.55)}

CONTRACTS = {"signed": (468, SIGNED_VALUES, SIGNED_START), "offer": (480, OFFER_VALUES, OFFER_START)}


def contract(rate: float = 0.07, *, months: int = 468, **changes) -> RuerupPension:
    return RuerupPension(
        monthly_contribution=500.0,
        contribution_months=months,
        yearly_return_rate=rate,
        **changes,
    )


def single_household(monthly_gross: float = 200_000 / 12.0) -> GermanHousehold:
    # 200k gross is deep in the 42% zone with the Soli fully phased in.
    return GermanHousehold(earners=(Earner(monthly_gross=monthly_gross),), joint=False)


def run(*components, years: int, household: GermanHousehold | None = None) -> pd.DataFrame:
    parts = [household or single_household(), *components, Etf(yearly_return_rate=0.0)]
    return FinancialModel(components=parts, years=years).run().set_index("year")


class TestTheDocuments:
    @pytest.mark.parametrize("name", sorted(CONTRACTS))
    @pytest.mark.parametrize("rate", [0.00, 0.03, 0.07, 0.09])
    def test_every_anniversary_value(self, name, rate):
        months, tables, _ = CONTRACTS[name]
        values = contract(rate, months=months).contract_values()
        for year, expected in enumerate(tables[rate], start=1):
            assert values[12 * year - 1] == pytest.approx(expected, rel=1.5e-4, abs=20), year

    @pytest.mark.parametrize("name", sorted(CONTRACTS))
    @pytest.mark.parametrize("rate", [0.00, 0.03, 0.07, 0.09])
    def test_the_capital_at_rentenbeginn(self, name, rate):
        months, _, starts = CONTRACTS[name]
        capital, _ = starts[rate]
        assert contract(rate, months=months).capital_at_pension_start() == pytest.approx(
            capital, rel=1e-4
        )

    @pytest.mark.parametrize("name", sorted(CONTRACTS))
    @pytest.mark.parametrize("rate", [0.00, 0.03, 0.07, 0.09])
    def test_the_pension_from_the_insurers_capital(self, name, rate):
        _, _, starts = CONTRACTS[name]
        capital, pension = starts[rate]
        annuity = DECLARED_LOWER if rate == 0.0 else DECLARED_2024
        assert annuity.monthly_pension(capital) == pytest.approx(pension, abs=0.02)

    def test_only_the_guaranteed_factor_is_the_default(self):
        assert RuerupAnnuity().monthly_pension(10_000.0) == pytest.approx(21.24683)
        # The signed contract at 7%: what is promised against what is shown.
        capital = SIGNED_START[0.07][0]
        assert RuerupAnnuity().monthly_pension(capital) == pytest.approx(1_976.86, abs=0.01)

    def test_the_acquisition_cost_matches_the_pib(self):
        signed = contract(months=468)
        assert signed.acquisition_total == pytest.approx(5_850.0)
        assert signed.monthly_acquisition == pytest.approx(97.5)
        assert contract(months=480).acquisition_total == pytest.approx(6_000.0)

    def test_the_pib_example_pins_the_funds_own_costs(self):
        # The PIB quotes gross returns, before the funds' costs; ~0.38% a year
        # of fund costs on top of the tariff reproduces its four examples.
        # The rest is the PIB's standardized method.
        fund_costs = 0.0038
        for gross, capital in ((0.06, 664_458), (0.05, 526_598), (0.02, 275_541), (0.0, 187_650)):
            rate = (1.0 + gross) * (1.0 - fund_costs) - 1.0
            assert contract(rate).capital_at_pension_start() == pytest.approx(capital, rel=4e-3)


class TestTheContract:
    def test_acquisition_cost_stops_after_five_years(self):
        values = contract(0.0, costs=RuerupCosts(capital_rate=0.0)).contract_values()
        assert values[0] == pytest.approx(500 * 0.955 - 97.5)
        assert values[60] - values[59] == pytest.approx(500 * 0.955)

    def test_a_short_term_spreads_the_acquisition_over_the_whole_term(self):
        short = contract(months=24)
        assert short.monthly_acquisition == pytest.approx(0.025 * 500 * 24 / 24)

    def test_nothing_in_means_nothing_out(self):
        empty = RuerupPension(contribution_months=12, yearly_return_rate=0.05)
        assert empty.contract_values() == [0.0] * 12


class TestTheEngine:
    def test_the_engine_walks_the_pure_path(self):
        frame = run(contract(), years=5)
        values = contract().contract_values()
        for year in range(1, 6):
            assert frame.loc[year, "ruerup.contract_value"] == values[12 * year - 1]

    def test_a_rebuilt_history_is_the_pure_path(self):
        frame = run(contract(elapsed_months=14), years=1)
        assert frame.loc[0, "ruerup.contract_value"] == contract().contract_values()[13]

    def test_a_split_run_is_bit_identical(self):
        whole = run(contract(), years=4)
        pinned = contract(elapsed_months=12, initial_value=whole.loc[1, "ruerup.contract_value"])
        split = run(pinned, years=3)
        for year in (1, 2, 3):
            assert split.loc[year, "ruerup.contract_value"] == whole.loc[year + 1, "ruerup.contract_value"]

    def test_rentenbeginn_on_the_real_calendar(self):
        # Model year 1 is 2026, 14 months into the contract: the last
        # contribution is October 2063 (year 38), the first pension November.
        annuity = RuerupAnnuity(factor=24.996271, bonus_rate=0.0187, yearly_increase_rate=0.01)
        frame = run(contract(elapsed_months=14, annuity=annuity), years=39)
        pension = frame.loc[38, "ruerup.monthly_pension"]
        assert frame.loc[37, "ruerup.monthly_pension"] == 0.0
        assert frame.loc[38, "ruerup.yearly_contributions"] == 10 * 500.0
        assert frame.loc[38, "ruerup.contract_value"] == 0.0
        assert frame.loc[38, "ruerup.yearly_pension"] == pytest.approx(2 * pension)
        assert pension == pytest.approx(annuity.monthly_pension(frame.loc[38, "ruerup.pension_capital"]))
        # The first increase is on the anniversary, in November of year 39.
        assert frame.loc[39, "ruerup.yearly_pension"] == pytest.approx(10 * pension + 2 * pension * 1.01)

    def test_the_converted_capital_is_the_pure_one(self):
        frame = run(contract(months=24), years=3)
        assert frame.loc[3, "ruerup.pension_capital"] == contract(months=24).capital_at_pension_start()

    def test_the_contribution_leaves_the_sink(self):
        alone = run(years=1)
        frame = run(contract(0.0), years=1)
        # The leftover is the household's; the contribution is withheld from it.
        assert frame.loc[1, "monthly_leftover"] == alone.loc[1, "monthly_leftover"]
        assert frame.loc[1, "monthly_invested"] == pytest.approx(frame.loc[1, "monthly_leftover"])
        assert frame.loc[1, "etf_capital"] == pytest.approx(alone.loc[1, "etf_capital"] - 6_000.0)
        assert frame.loc[1, "estimated_total_capital"] == pytest.approx(
            alone.loc[1, "estimated_total_capital"] - 6_000.0
        )

    def test_the_pension_is_cash(self):
        frame = run(contract(0.0, months=12, annuity=DECLARED_2024), years=3)
        alone = run(years=3)
        received = frame.loc[3, "etf_capital"] - frame.loc[2, "etf_capital"]
        baseline = alone.loc[3, "etf_capital"] - alone.loc[2, "etf_capital"]
        # Year 3 also carries the July settlement of year 2's pension tax.
        refund_gap = (
            frame.loc[3, "household.yearly_refund_received"]
            - alone.loc[3, "household.yearly_refund_received"]
        )
        assert received - baseline == pytest.approx(frame.loc[3, "ruerup.yearly_pension"] + refund_gap)

    def test_component_order_does_not_change_the_result(self):
        parts = [single_household(), contract(elapsed_months=440), Etf(yearly_return_rate=0.05)]
        forward = FinancialModel(components=parts, years=5).run()
        backward = FinancialModel(components=list(reversed(parts)), years=5).run()
        pd.testing.assert_frame_equal(forward, backward[forward.columns], check_exact=True)

    def test_a_rise_is_a_second_contract_ending_in_the_same_month(self):
        rise = RuerupPension(
            monthly_contribution=250.0, contribution_months=454, yearly_return_rate=0.07, name="rise"
        )
        frame = run(contract(elapsed_months=14), rise, years=38)
        assert frame.loc[1, "household.yearly_basisrente_deduction"] == pytest.approx(9_000.0)
        for name in ("ruerup", "rise"):
            assert frame.loc[37, f"{name}.pension_capital"] == 0.0
            assert frame.loc[38, f"{name}.pension_capital"] > 0.0
        assert rise.acquisition_total == pytest.approx(0.025 * 250 * 454)

    def test_the_model_needs_a_household(self):
        with pytest.raises(ValueError, match="GermanHousehold"):
            FinancialModel(
                components=[Salary(monthly_amount=5_000), contract(), Etf()], years=1
            )


@pytest.fixture(scope="module")
def pair():
    """The 200k single earner without and with the signed contract."""
    return run(years=3), run(contract(), years=3)


class TestTheTaxes:
    def test_payroll_ignores_the_contribution(self, pair):
        alone, frame = pair
        for year in (1, 2, 3):
            assert frame.loc[year, "household.yearly_withheld_tax"] == alone.loc[year, "household.yearly_withheld_tax"]

    def test_the_assessment_deducts_the_contribution(self, pair):
        _, frame = pair
        social = social_contributions(200_000.0, SYSTEM, n_children=0, private_health=False)
        figures = [EarnerFigures(yearly_gross=200_000.0, social=social)]
        expected = assessed_tax(
            taxable_income(figures, SYSTEM, joint=False, basisrente_contributions=6_000.0),
            SYSTEM,
            joint=False,
            n_children=0,
        )
        assert frame.loc[1, "household.yearly_assessed_tax"] == expected.total
        assert frame.loc[1, "household.yearly_basisrente_deduction"] == 6_000.0

    def test_the_saving_arrives_with_next_julys_refund(self, pair):
        alone, frame = pair
        saved = alone.loc[1, "household.yearly_assessed_tax"] - frame.loc[1, "household.yearly_assessed_tax"]
        assert frame.loc[1, "household.yearly_refund_received"] == alone.loc[1, "household.yearly_refund_received"]
        gained = frame.loc[2, "household.yearly_refund_received"] - alone.loc[2, "household.yearly_refund_received"]
        assert gained == saved
        # Deep in the 42% zone with the full Soli: 44.31 cents per euro.
        assert saved == pytest.approx(0.42 * 1.055 * 6_000.0, abs=3.0)

    def test_the_cap_leaves_room_only_above_the_statutory_contributions(self):
        social = social_contributions(200_000.0, SYSTEM, n_children=0, private_health=False)
        room = basisrente_cap(SYSTEM, joint=False) - 2.0 * social.pension
        assert room == pytest.approx(30_826.0 - 18_860.4)
        big = RuerupPension(monthly_contribution=2_000.0, contribution_months=120)
        frame = run(big, years=1)
        assert frame.loc[1, "household.yearly_basisrente_deduction"] == pytest.approx(room)
        bigger = run(RuerupPension(monthly_contribution=3_000.0, contribution_months=120), years=1)
        assert bigger.loc[1, "household.yearly_assessed_tax"] == frame.loc[1, "household.yearly_assessed_tax"]

    def test_a_couple_shares_a_doubled_cap(self):
        couple = GermanHousehold(
            earners=(Earner(monthly_gross=200_000 / 12.0), Earner(monthly_gross=0.0))
        )
        frame = run(RuerupPension(monthly_contribution=4_000.0, contribution_months=120), years=1, household=couple)
        social = social_contributions(200_000.0, SYSTEM, n_children=0, private_health=False)
        assert frame.loc[1, "household.yearly_basisrente_deduction"] == pytest.approx(
            61_652.0 - 2.0 * social.pension
        )

    def test_the_pension_is_assessed_and_settled_a_year_late(self):
        retired = GermanHousehold(earners=(Earner(monthly_gross=0.0),), joint=False)
        payout = contract(0.0, elapsed_months=468, initial_value=600_000.0, annuity=DECLARED_2024)
        frame = run(payout, years=2, household=retired)
        pension = frame.loc[1, "ruerup.yearly_pension"]
        assert pension == pytest.approx(12 * DECLARED_2024.monthly_pension(600_000.0))
        assert frame.loc[1, "household.yearly_pension_income"] == pension
        nobody = EarnerFigures(yearly_gross=0.0, social=SocialContributions(0.0, 0.0, 0.0, 0.0))
        expected = assessed_tax(
            taxable_income([nobody], SYSTEM, joint=False, pension_income=pension),
            SYSTEM,
            joint=False,
            n_children=0,
        )
        assert frame.loc[1, "household.yearly_assessed_tax"] == expected.total > 0.0
        assert frame.loc[2, "household.yearly_refund_received"] == -expected.total

    def test_a_contract_without_contributions_changes_nothing(self):
        alone = run(years=2)
        empty = run(RuerupPension(contribution_months=120), years=2)
        for column in ("household.yearly_assessed_tax", "household.yearly_refund_received", "etf_capital"):
            assert (empty[column] == alone[column]).all()


class TestValidation:
    def test_the_term_is_checked(self):
        with pytest.raises(ValueError, match="contribution_months"):
            RuerupPension(monthly_contribution=500.0)
        with pytest.raises(TypeError, match="contribution_months"):
            RuerupPension(contribution_months=12.0)
        with pytest.raises(ValueError, match="elapsed_months"):
            contract(months=12, elapsed_months=13)
        with pytest.raises(ValueError, match="elapsed_months"):
            contract(elapsed_months=-1)

    def test_amounts_and_rates_are_checked(self):
        with pytest.raises(ValueError, match="monthly_contribution"):
            RuerupPension(monthly_contribution=-1.0, contribution_months=12)
        with pytest.raises(ValueError, match="initial_value"):
            contract(initial_value=-1.0)
        with pytest.raises(ValueError, match="yearly_return_rate"):
            contract(-1.0)
        with pytest.raises(ValueError, match="capital_rate"):
            RuerupCosts(capital_rate=1.0)
        with pytest.raises(ValueError, match="acquisition_months"):
            RuerupCosts(acquisition_months=0)
        with pytest.raises(ValueError, match="factor"):
            RuerupAnnuity(factor=0.0)
        with pytest.raises(TypeError, match="costs"):
            contract(costs=0.004)
        with pytest.raises(TypeError, match="annuity"):
            contract(annuity=21.25)
