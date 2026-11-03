"""Buy-to-let taxation against two reference calculations.

The expected figures are hardcoded from two calculator sheets (kept out of
git): `data/KL_D12_Kalkulation.pdf` (Kaiserslautern, 510k, AfA 3%) and
`data/Pirmasens_ETW5_Kalkulation.pdf` (180k, AfA 2%). Like
`test_german_tax.py`, this file is an independent check against published
numbers, not a snapshot of the model's own output.

Two convention gaps separate the model from the sheets, and every loose
tolerance below is sized by one of them:

- The sheets amortize the loan in yearly steps (a year's interest is the rate
  on the year-start balance, the principal is applied at year end); the model
  amortizes monthly. Monthly repayment starts eating the principal earlier,
  so the model pays ~0.7% less interest in year 1, ~1.9% less in year 11, and
  ends year 11 with a ~0.68% smaller Restschuld.
- The sheets tax the rental result at a flat Grenzsteuersatz (42% and 40%);
  the model runs the real progressive assessment through `GermanHousehold`.
  The test household is a single earner deep in the flat 42% zone with the
  Soli fully phased in, so its true marginal rate is 0.42 * 1.055 = 0.4431 —
  constant across the horizon (the zvE stays inside the drifting zone) and
  exact up to the euro flooring of the tariff. A stable 40% household cannot
  exist under the progressive tariff, so both sheets are checked against this
  42% household and the sheets' taxable results are re-rated at 0.4431.
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
    Salary,
    payoff_year,
)
from src.model.german_tax import (
    EarnerFigures,
    TaxSystem,
    assessed_tax,
    social_contributions,
    taxable_income,
)

MARGINAL = 0.42 * 1.055  # top-zone rate with full Solidaritaetszuschlag


def household() -> GermanHousehold:
    # 200k gross -> zvE 181,770.80: flat 42% zone (ends at 277,826) with the
    # Soli far beyond its Milderungszone.
    return GermanHousehold(earners=(Earner(monthly_gross=200_000 / 12.0),), joint=False)


def kl_d12(*, taxed: bool = True) -> InvestmentProperty:
    """Kaiserslautern D12: 206 m2, new build, 100% financing."""
    return InvestmentProperty(
        price=510_000,
        yearly_price_increase_rate=0.02,
        monthly_rental_income=2_820,
        yearly_rental_income_increase_rate=0.02,
        mortgage=Mortgage(
            interest_rate=0.045,
            down_payment_rate=0.0,
            fees_rate=0.12,
            yearly_repayment_rate=0.015,
        ),
        operating_costs=OperatingCosts(
            monthly_deductible=150.0,  # non-recoverable Hausgeld
            monthly_own_reserve=206.0,  # 12 EUR/m2 a year on 206 m2
            yearly_increase_rate=0.02,
        ),
        taxation=RentalTaxation(afa_rate=0.03, building_share=0.96) if taxed else None,
        name="flat",
    )


def pirmasens(*, taxed: bool = True) -> InvestmentProperty:
    """Pirmasens ETW 5: 113 m2, existing building, 100% financing."""
    return InvestmentProperty(
        price=180_000,
        yearly_price_increase_rate=0.02,
        monthly_rental_income=1_050,
        yearly_rental_income_increase_rate=0.02,
        mortgage=Mortgage(
            interest_rate=0.045,
            down_payment_rate=0.0,
            fees_rate=0.1037,
            yearly_repayment_rate=0.015,
        ),
        operating_costs=OperatingCosts(
            monthly_deductible=37.0,
            monthly_own_reserve=56.5,  # 6 EUR/m2 a year on 113 m2
            yearly_increase_rate=0.02,
        ),
        taxation=RentalTaxation(afa_rate=0.02, building_share=0.80) if taxed else None,
        name="flat",
    )


def run(flat: InvestmentProperty | None, *, years: int = 12):
    components = [household(), Etf(initial_capital=100_000, yearly_return_rate=0.0)]
    if flat is not None:
        components.insert(1, flat)
    return FinancialModel(components=components, years=years).run().set_index("year")


@pytest.fixture(scope="module")
def alone():
    return run(None)


@pytest.fixture(scope="module")
def kl(alone):
    return run(kl_d12())


@pytest.fixture(scope="module")
def pir(alone):
    return run(pirmasens())


def tax_delta(with_flat, alone, year: int) -> float:
    """Extra tax the flat causes: positive means a refund (a rental loss)."""
    return (
        alone.loc[year, "household.yearly_assessed_tax"]
        - with_flat.loc[year, "household.yearly_assessed_tax"]
    )


# --- exact figures: conventions identical to the sheets ---------------------


@pytest.mark.parametrize(
    "build, afa_base, monthly_afa, fees, payment",
    [
        # sheet: Basis fuer langfristige Abschreibung 548,352 = 571,200 * 96%
        (kl_d12, 548_352.0, 1_370.88, 61_200.0, 2_550.0),
        # sheet: 158,933 = 198,666 * 80% (the sheet rounds to whole euros)
        (pirmasens, 158_932.80, 264.888, 18_666.0, 900.0),
    ],
)
def test_afa_and_financing_derivations(build, afa_base, monthly_afa, fees, payment):
    flat = build()
    assert flat.afa_base == pytest.approx(afa_base, rel=1e-12)
    assert flat.monthly_afa == pytest.approx(monthly_afa, rel=1e-12)
    assert flat.purchase_fees == pytest.approx(fees, rel=1e-12)
    assert flat.initial_loan == pytest.approx(flat.price, rel=1e-12)  # 100% financing
    assert flat.initial_monthly_payment == pytest.approx(payment, rel=1e-12)


def test_initial_row_is_the_equity_outlay(kl, pir):
    # Only the purchase fees are paid from equity; the loan covers the price.
    assert kl.loc[0, "etf_capital"] == pytest.approx(100_000 - 61_200, abs=1e-6)
    assert kl.loc[0, "flat.cumulative_cashflow"] == pytest.approx(-61_200, abs=1e-6)
    assert kl.loc[0, "flat.monthly_operating_costs"] == pytest.approx(356.0, abs=1e-9)
    assert pir.loc[0, "etf_capital"] == pytest.approx(100_000 - 18_666, abs=1e-6)
    assert pir.loc[0, "flat.cumulative_cashflow"] == pytest.approx(-18_666, abs=1e-6)
    assert pir.loc[0, "flat.monthly_operating_costs"] == pytest.approx(93.5, abs=1e-9)


def test_rent_costs_and_value_trajectories(kl, pir):
    # Rent and costs escalate at year end: year 11 has seen ten increases.
    assert kl.loc[1, "flat.monthly_rental_income"] == pytest.approx(2_820.0, rel=1e-12)
    assert kl.loc[11, "flat.monthly_rental_income"] == pytest.approx(
        2_820 * 1.02**10, rel=1e-9
    )
    assert kl.loc[11, "flat.monthly_operating_costs"] == pytest.approx(
        356 * 1.02**10, rel=1e-9
    )
    assert pir.loc[11, "flat.monthly_operating_costs"] == pytest.approx(
        93.5 * 1.02**10, rel=1e-9
    )
    for year in (1, 6, 11):
        assert kl.loc[year, "flat.property_value"] == pytest.approx(
            510_000 * 1.02**year, rel=1e-12
        )
    # sheet: Wert der Immobilie (Jahresende) 2036 = 634,121
    assert kl.loc[11, "flat.property_value"] == pytest.approx(634_121, abs=1.0)
    assert pir.loc[11, "flat.property_value"] == pytest.approx(223_807, abs=1.0)
    # The Kapitaldienst never changes while the loan runs.
    for year in (1, 11):
        assert kl.loc[year, "flat.monthly_mortgage_payment"] == pytest.approx(
            2_550.0, rel=1e-12
        )
        assert pir.loc[year, "flat.monthly_mortgage_payment"] == pytest.approx(
            900.0, rel=1e-12
        )


def test_afa_is_constant_over_time(kl):
    for year in (0, 1, 6, 11):
        assert kl.loc[year, "flat.monthly_afa"] == pytest.approx(1_370.88, rel=1e-9)


def test_taxable_result_matches_its_closed_form(kl, pir):
    # The deductible part is not reported on its own; recompute its escalation.
    for frame, deductible0, monthly_afa in ((kl, 150.0, 1_370.88), (pir, 37.0, 264.888)):
        for year in (1, 6, 11):
            row = frame.loc[year]
            deductible = deductible0 * 1.02 ** (year - 1)
            expected = (
                12.0 * row["flat.monthly_rental_income"]
                - 12.0 * deductible
                - 12.0 * row["flat.monthly_interest_payment"]
                - 12.0 * monthly_afa
            )
            assert row["flat.yearly_taxable_result"] == pytest.approx(expected, rel=1e-9)


def test_household_reports_what_the_flat_published(kl, pir):
    for frame in (kl, pir):
        for year in (1, 6, 11):
            assert frame.loc[year, "household.yearly_rental_result"] == pytest.approx(
                frame.loc[year, "flat.yearly_taxable_result"], rel=1e-12
            )


def test_tax_delta_is_the_marginal_rate_on_the_result(kl, pir, alone):
    # The household sits in the flat 42% zone, so the assessment must move by
    # exactly the marginal rate times the rental result, up to euro flooring.
    for frame in (kl, pir):
        for year in (1, 11):
            expected = -frame.loc[year, "flat.yearly_taxable_result"] * MARGINAL
            assert tax_delta(frame, alone, year) == pytest.approx(expected, abs=5.0)
    # Signs: KL starts as a loss (refund), Pirmasens as a profit (back-payment).
    assert tax_delta(kl, alone, 1) > 0
    assert tax_delta(pir, alone, 1) < 0


# --- sheet anchors: loose, sized by the amortization-convention gap ---------


@pytest.mark.parametrize(
    "fixture, interest_y1, interest_y11, loan_y11",
    [
        # sheet monthly figures x12: Zinsen 1,912.50 / 1,560; Restschuld 404,115
        ("kl", 22_950.0, 18_720.0, 404_115.0),
        # sheet: Zinsen 675 / 551; Restschuld 142,629
        ("pir", 8_100.0, 6_607.0, 142_629.0),
    ],
)
def test_loan_against_sheet(fixture, interest_y1, interest_y11, loan_y11, request):
    frame = request.getfixturevalue(fixture)
    assert 12.0 * frame.loc[1, "flat.monthly_interest_payment"] == pytest.approx(
        interest_y1, rel=0.01
    )
    assert 12.0 * frame.loc[11, "flat.monthly_interest_payment"] == pytest.approx(
        interest_y11, rel=0.025
    )
    assert frame.loc[11, "flat.total_loan"] == pytest.approx(loan_y11, rel=0.01)


def test_payoff_year_against_sheet():
    # The sheets reach Volltilgung in 2057 — their year 32. Monthly
    # amortization repays in month 371, one model year earlier.
    for build in (kl_d12, pirmasens):
        assert payoff_year(run(build(), years=35).reset_index()) == 31


@pytest.mark.parametrize(
    "fixture, taxable_y1, taxable_y11, tol_y1, tol_y11",
    [
        # sheet Nebenrechnung x12: -613.38 x 12; +324 x 12
        ("kl", -7_360.56, 3_888.0, 170.0, 400.0),
        # sheet: +73.11 x 12; +419.33 x 12
        ("pir", 877.34, 5_032.0, 60.0, 150.0),
    ],
)
def test_taxable_result_against_sheet(fixture, taxable_y1, taxable_y11, tol_y1, tol_y11, request):
    frame = request.getfixturevalue(fixture)
    assert frame.loc[1, "flat.yearly_taxable_result"] == pytest.approx(
        taxable_y1, abs=tol_y1
    )
    assert frame.loc[11, "flat.yearly_taxable_result"] == pytest.approx(
        taxable_y11, abs=tol_y11
    )


def test_tax_effect_against_sheet(kl, pir, alone):
    # Sheet taxable results re-rated at this household's true marginal rate.
    assert tax_delta(kl, alone, 1) == pytest.approx(7_360.56 * MARGINAL, abs=100.0)
    assert tax_delta(kl, alone, 11) == pytest.approx(-3_888.0 * MARGINAL, abs=200.0)
    assert tax_delta(pir, alone, 1) == pytest.approx(-877.34 * MARGINAL, abs=40.0)
    assert tax_delta(pir, alone, 11) == pytest.approx(-5_032.0 * MARGINAL, abs=100.0)


# --- cash timing ------------------------------------------------------------


def test_rental_tax_settles_as_july_cash_one_year_later(kl, alone):
    # Year 1 cash carries no tax effect at all (0% portfolio, so the ETF
    # difference is exactly the flat's own cash flow).
    assert kl.loc[1, "etf_capital"] - alone.loc[1, "etf_capital"] == pytest.approx(
        -61_200 + 12.0 * (2_820 - 356 - 2_550), abs=1e-6
    )
    # The year-1 assessment difference arrives with the July refund of year 2.
    refund_diff = (
        kl.loc[2, "household.yearly_refund_received"]
        - alone.loc[2, "household.yearly_refund_received"]
    )
    assert refund_diff == pytest.approx(tax_delta(kl, alone, 1), abs=1e-6)
    year2_flat_cash = 12.0 * (
        kl.loc[2, "flat.monthly_rental_income"]
        - kl.loc[2, "flat.monthly_operating_costs"]
        - 2_550.0
    )
    expected = (
        kl.loc[1, "etf_capital"]
        - alone.loc[1, "etf_capital"]
        + year2_flat_cash
        + tax_delta(kl, alone, 1)
    )
    assert kl.loc[2, "etf_capital"] - alone.loc[2, "etf_capital"] == pytest.approx(
        expected, abs=0.01
    )


def test_cumulative_cashflow_tracks_the_operative_cash(kl):
    # -EK, then + rent - costs - Kapitaldienst, year after year (pre-tax).
    expected = -61_200.0
    for year in (1, 2, 3):
        expected += 12.0 * (
            kl.loc[year, "flat.monthly_rental_income"]
            - kl.loc[year, "flat.monthly_operating_costs"]
            - kl.loc[year, "flat.monthly_mortgage_payment"]
        )
        assert kl.loc[year, "flat.cumulative_cashflow"] == pytest.approx(expected, rel=1e-9)


# --- semantics --------------------------------------------------------------


def test_vacancy_scales_cash_and_taxable_income(kl):
    vacant = run(replace(kl_d12(), vacancy_rate=0.1))
    assert vacant.loc[1, "flat.monthly_rental_income"] == pytest.approx(
        2_820 * 0.9, rel=1e-12
    )
    # The lost rent leaves both the cash and the taxable result.
    assert vacant.loc[1, "flat.yearly_taxable_result"] == pytest.approx(
        kl.loc[1, "flat.yearly_taxable_result"] - 12.0 * 282.0, rel=1e-9
    )
    assert vacant.loc[1, "etf_capital"] - kl.loc[1, "etf_capital"] == pytest.approx(
        -12.0 * 282.0, abs=1e-6
    )


def test_vacancy_without_taxation_needs_no_household():
    def build(vacancy: float):
        return FinancialModel(
            components=[
                Salary(monthly_amount=3_000),
                replace(kl_d12(taxed=False), vacancy_rate=vacancy),
                Etf(initial_capital=100_000, yearly_return_rate=0.0),
            ],
            years=2,
        ).run().set_index("year")

    full, vacant = build(0.0), build(0.1)
    assert vacant.loc[1, "flat.monthly_rental_income"] == pytest.approx(
        2_820 * 0.9, rel=1e-12
    )
    assert vacant.loc[1, "etf_capital"] - full.loc[1, "etf_capital"] == pytest.approx(
        -12.0 * 282.0, abs=1e-6
    )
    assert vacant.loc[1, "flat.yearly_taxable_result"] == 0.0


def test_untaxed_flat_leaves_the_household_alone(alone):
    plain = run(kl_d12(taxed=False))
    for year in (1, 6, 11):
        # Bit-identical: the wage-only assessment path must not even be re-run.
        assert (
            plain.loc[year, "household.yearly_assessed_tax"]
            == alone.loc[year, "household.yearly_assessed_tax"]
        )
        assert plain.loc[year, "household.yearly_rental_result"] == 0.0
        assert plain.loc[year, "flat.yearly_taxable_result"] == 0.0


def test_two_flats_add_their_results(kl, pir, alone):
    both = FinancialModel(
        components=[
            household(),
            replace(kl_d12(), name="flat1"),
            replace(pirmasens(), name="flat2"),
            Etf(initial_capital=150_000, yearly_return_rate=0.0),
        ],
        years=12,
    ).run().set_index("year")
    for year in (1, 11):
        assert both.loc[year, "household.yearly_rental_result"] == pytest.approx(
            both.loc[year, "flat1.yearly_taxable_result"]
            + both.loc[year, "flat2.yearly_taxable_result"],
            rel=1e-12,
        )
        assert tax_delta(both, alone, year) == pytest.approx(
            -(
                both.loc[year, "flat1.yearly_taxable_result"]
                + both.loc[year, "flat2.yearly_taxable_result"]
            )
            * MARGINAL,
            abs=5.0,
        )


def test_losses_beyond_the_zve_are_forfeited():
    # Year 1: a small salary and a huge AfA — the assessment bottoms out at
    # zero and the excess loss is gone. Year 2: a five-fold raise — with a
    # Verlustvortrag the carried ~40k loss would wipe most of that year's
    # tax too; without one the year is taxed on its own result only.
    frame = FinancialModel(
        components=[
            GermanHousehold(
                earners=(Earner(monthly_gross=2_000, yearly_increase_rate=4.0),),
                joint=False,
            ),
            InvestmentProperty(
                price=2_000_000,
                monthly_rental_income=0.0,
                taxation=RentalTaxation(afa_rate=0.03, building_share=1.0),
                name="flat",
            ),
            Etf(initial_capital=2_100_000, yearly_return_rate=0.0),
        ],
        years=2,
    ).run().set_index("year")
    assert frame.loc[1, "flat.yearly_taxable_result"] == pytest.approx(-60_000, rel=1e-12)
    assert frame.loc[1, "household.yearly_assessed_tax"] == 0.0
    # The July refund of year 2 returns everything that was withheld in year 1.
    assert frame.loc[2, "household.yearly_refund_received"] == pytest.approx(
        frame.loc[1, "household.yearly_withheld_tax"], abs=1e-6
    )
    # Forfeited, not carried: 120k gross minus the year's own 60k loss still
    # leaves a clearly positive assessment.
    assert frame.loc[2, "household.yearly_assessed_tax"] > 5_000.0


def test_joint_household_with_children_assesses_the_rental_result():
    # Exercises the joint=/n_children= pass-through in year_assess against an
    # independent recomputation with the pure german_tax functions
    # (Ehegattensplitting and the Kindergeld Guenstigerpruefung included).
    system = TaxSystem()
    earners = (Earner(monthly_gross=5_000), Earner(monthly_gross=3_000))
    frame = FinancialModel(
        components=[
            GermanHousehold(earners=earners, joint=True, tax_classes="IV/IV", n_children=2),
            InvestmentProperty(
                price=200_000,
                monthly_rental_income=0.0,
                taxation=RentalTaxation(afa_rate=0.03, building_share=1.0),
                name="flat",
            ),
            Etf(initial_capital=250_000, yearly_return_rate=0.0),
        ],
        years=2,
    ).run().set_index("year")
    rental = frame.loc[1, "household.yearly_rental_result"]
    assert rental == pytest.approx(-6_000.0, rel=1e-12)  # cash purchase: pure AfA
    figures = [
        EarnerFigures(
            yearly_gross=earner.monthly_gross * 12.0,
            social=social_contributions(
                earner.monthly_gross * 12.0, system, n_children=2, private_health=False
            ),
        )
        for earner in earners
    ]
    expected = assessed_tax(
        taxable_income(figures, system, joint=True, other_income=rental),
        system,
        joint=True,
        n_children=2,
    ).total
    assert frame.loc[1, "household.yearly_assessed_tax"] == pytest.approx(expected, abs=1e-9)


def test_taxation_composes_with_sondertilgung_and_refinancing():
    flat = replace(
        kl_d12(),
        mortgage=Mortgage(
            interest_rate=0.045,
            down_payment_rate=0.0,
            fees_rate=0.12,
            yearly_repayment_rate=0.015,
            refinancing_years=5,
            sondertilgung_yearly_rate=0.02,
        ),
    )
    frame = run(flat)
    sonder = frame.loc[1, "flat.yearly_sondertilgung"]
    assert sonder == pytest.approx(0.02 * 510_000, rel=1e-9)
    # The taxable result keeps its closed form with the actual (now lower)
    # interest, across the year-5 refinancing.
    for year in (1, 6):
        row = frame.loc[year]
        expected = (
            12.0 * row["flat.monthly_rental_income"]
            - 12.0 * 150.0 * 1.02 ** (year - 1)
            - 12.0 * row["flat.monthly_interest_payment"]
            - 12.0 * 1_370.88
        )
        assert row["flat.yearly_taxable_result"] == pytest.approx(expected, rel=1e-9)
    # The withheld Sondertilgung is part of the flat's own cash outflow.
    expected_cumulative = -61_200.0 + 12.0 * (2_820.0 - 356.0 - 2_550.0) - sonder
    assert frame.loc[1, "flat.cumulative_cashflow"] == pytest.approx(
        expected_cumulative, rel=1e-9
    )


def test_a_profit_is_asked_for_in_advance(pir, kl):
    # Pirmasens's profit leaves a back-payment every year, so the Finanzamt
    # sets Vorauszahlungen from year 2 on. It raises them only once the raise
    # reaches 100 EUR a date (§37 Abs. 5 S. 2 EStG): years 3 and 4 keep year
    # 2's rate though the back-payment grows, year 5 catches up.
    shortfall = pir["household.yearly_assessed_tax"] - pir["household.yearly_withheld_tax"]
    prepaid = pir["household.yearly_prepaid_tax"]
    assert prepaid[1] == 0.0
    assert prepaid[2] == pytest.approx(shortfall[1])
    for year in (3, 4):
        assert shortfall[year - 1] / 4.0 - shortfall[1] / 4.0 < 100.0
        assert prepaid[year] == pytest.approx(shortfall[1])
    assert shortfall[4] / 4.0 - shortfall[1] / 4.0 >= 100.0
    assert prepaid[5] == pytest.approx(shortfall[4])
    # Each July settlement credits what was paid in advance.
    for year in range(2, 12):
        assert pir.loc[year + 1, "household.yearly_refund_received"] == pytest.approx(
            prepaid[year] - shortfall[year]
        )
    # KL's loss brings refunds, which set none — until its result turns.
    assert (kl.loc[:9, "household.yearly_prepaid_tax"] == 0.0).all()
    assert kl.loc[10, "household.yearly_prepaid_tax"] > 0.0


def test_final_year_settlement_is_dropped_beyond_the_horizon(pir, alone):
    # The last simulated year's assessment never becomes cash: its July
    # settlement lies beyond the horizon. For this profitable flat that is a
    # dropped back-payment — the part of the year's rental tax that its
    # Vorauszahlungen did not cover — so the final wealth is *overstated* by
    # it, the inverse of the conservative wage-refund cut.
    short = run(pirmasens(), years=2)
    operative = -18_666.0
    for year in (1, 2):
        operative += 12.0 * (
            short.loc[year, "flat.monthly_rental_income"]
            - short.loc[year, "flat.monthly_operating_costs"]
            - 900.0
        )
    # Year 1's back-payment set Vorauszahlungen, paid on two dates of year 2.
    prepaid = short.loc[2, "household.yearly_prepaid_tax"] - alone.loc[2, "household.yearly_prepaid_tax"]
    assert prepaid > 0.0
    diff = short.loc[2, "etf_capital"] - alone.loc[2, "etf_capital"]
    assert diff == pytest.approx(operative + tax_delta(short, alone, 1) - prepaid, abs=0.01)
    # The dropped year-2 settlement is still material, so the omission matters.
    assert abs(tax_delta(short, alone, 2) + prepaid) > 100.0


# --- validation -------------------------------------------------------------


def test_taxation_requires_a_household():
    with pytest.raises(ValueError, match="GermanHousehold"):
        FinancialModel(
            components=[kl_d12(), Etf(initial_capital=100_000)],
            years=2,
        )


def test_operating_costs_alone_need_no_household():
    FinancialModel(
        components=[kl_d12(taxed=False), Etf(initial_capital=100_000)],
        years=2,
    ).run()


def test_parameter_validation():
    with pytest.raises(ValueError, match="building_share"):
        RentalTaxation(afa_rate=0.02, building_share=1.5)
    with pytest.raises(ValueError, match="afa_rate"):
        RentalTaxation(afa_rate=1.5, building_share=0.8)
    with pytest.raises(ValueError, match="monthly_deductible"):
        OperatingCosts(monthly_deductible=-1.0)
    with pytest.raises(ValueError, match="vacancy_rate"):
        InvestmentProperty(price=100_000, vacancy_rate=1.5)
    with pytest.raises(ValueError, match="vacancy_rate"):
        InvestmentProperty(price=100_000, vacancy_rate=-0.1)
    with pytest.raises(TypeError, match="operating_costs"):
        InvestmentProperty(price=100_000, operating_costs=123)
    with pytest.raises(TypeError, match="taxation"):
        InvestmentProperty(price=100_000, taxation="yes")
