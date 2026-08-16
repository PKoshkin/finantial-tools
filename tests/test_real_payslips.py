"""The German payroll chain against real payslips and insurance contracts.

Sources (personal documents, deliberately kept out of git — `/data/` is
ignored; the figures below are hardcoded from them):

- DATEV payslip May 2026 (`data/pk_payslip-May-2026.pdf`): Steuerklasse IV,
  no children on the tax card, no church tax, privately insured. Monthly:
  gross 8,333.33 (contract salary 100,000/yr), Lohnsteuer 2,135.66, Soli
  52.34, RV 775.00, AV 108.33, no KV/PV contributions, employer subsidies
  319.00 (KV) + 28.50 (PV), Auszahlungsbetrag 5,609.50.
- HanseMerkur ELStAM letter of 03.12.2025 for Beitragsjahr 2026: full
  premiums KV 638.00 + PPV 57.00 a month; deductible Basisabsicherung
  (§10 Abs. 1 Nr. 3) 349.00 + 57.00 a month.
- Actual insurer debit 512.05 a month: the full 695.00 minus the 182.95
  Pauschalerstattung taken as a Beitragsstundung (2024 anchor of the same
  contract: 674.34 - 181.93 = 492.41).

Rounding conventions the tolerances stem from: the payroll floors the
yearly tax to full euros and the monthly figure to full cents, while the
model keeps the exact yearly figure and divides by 12 — so monthly amounts
may differ by up to 2 cents, never more.

The payslip only shows the *withholding* side; the yearly assessment and
the July refund have no counterpart on it. Once a second earner's payslip
is available, extend `CASES` and build the real two-earner IV/IV household
here; a Steuerbescheid would then pin the assessment side too.
"""

from dataclasses import dataclass, replace

import pytest

from src.model import Earner, GermanHousehold, PrivateHealthInsurance
from src.model.german_tax import (
    TaxSystem,
    private_deductible,
    private_subsidy,
    social_contributions,
    withholding,
)

SYSTEM = TaxSystem()


@dataclass(frozen=True)
class PayslipCase:
    """One real payslip month plus the insurance contract behind it."""

    label: str
    monthly_gross: float  # the contract salary; the slip shows it rounded
    tax_class: str
    n_children: int
    private_health: PrivateHealthInsurance | None
    expected_deductible_yearly: float  # ELStAM basis x 12 minus the subsidy
    # expected monthly figures, straight from the payslip:
    lohnsteuer: float
    soli: float
    rv: float
    av: float
    ag_zuschuss_kv: float
    ag_zuschuss_pv: float
    auszahlung: float
    insurer_debit: float  # what the insurer actually debits per month


PAVEL_MAY_2026 = PayslipCase(
    label="pavel_2026_05",
    monthly_gross=100_000 / 12.0,
    tax_class="IV",
    n_children=0,
    private_health=PrivateHealthInsurance(
        monthly_premium=638.0,
        monthly_care_premium=57.0,
        deductible_share=349.0 / 638.0,  # ELStAM: 349 of the 638 is Basis
        monthly_rebate=182.95,  # Beitragsstundung: 695.00 - 512.05
    ),
    expected_deductible_yearly=702.0,  # (349 + 57) x 12 - 12 x 347.50
    lohnsteuer=2_135.66,
    soli=52.34,
    rv=775.00,
    av=108.33,
    ag_zuschuss_kv=319.00,
    ag_zuschuss_pv=28.50,
    auszahlung=5_609.50,
    insurer_debit=512.05,
)

CASES = [PAVEL_MAY_2026]  # extend with the second earner's months as they arrive

parametrized = pytest.mark.parametrize("case", CASES, ids=lambda case: case.label)


@parametrized
def test_contract_and_debit_are_consistent(case):
    # The case data itself must reconcile: full premiums minus the rebate is
    # what the insurer debits.
    contract = case.private_health
    assert (
        contract.monthly_premium + contract.monthly_care_premium - contract.monthly_rebate
    ) == pytest.approx(case.insurer_debit, abs=0.005)


@parametrized
def test_social_contributions_match_the_payslip(case):
    social = social_contributions(
        case.monthly_gross * 12.0,
        SYSTEM,
        n_children=case.n_children,
        private_health=case.private_health is not None,
    )
    assert social.pension / 12.0 == pytest.approx(case.rv, abs=0.01)
    assert social.unemployment / 12.0 == pytest.approx(case.av, abs=0.01)
    if case.private_health is not None:
        assert social.health == 0.0
        assert social.care == 0.0


@parametrized
def test_employer_subsidy_matches_the_payslip(case):
    contract = case.private_health
    subsidy = private_subsidy(
        contract.monthly_premium, contract.monthly_care_premium, SYSTEM
    )
    assert subsidy == pytest.approx(case.ag_zuschuss_kv + case.ag_zuschuss_pv, abs=0.005)
    # Both halves stay under their statutory caps, so the subsidy is exactly
    # half of each premium — which is what the payslip shows line by line.
    assert 0.5 * contract.monthly_premium == pytest.approx(case.ag_zuschuss_kv, abs=0.005)
    assert 0.5 * contract.monthly_care_premium == pytest.approx(
        case.ag_zuschuss_pv, abs=0.005
    )


@parametrized
def test_deductible_premium_matches_the_elstam_letter(case):
    contract = case.private_health
    deductible = private_deductible(
        contract.monthly_premium,
        contract.monthly_care_premium,
        contract.deductible_share,
        SYSTEM,
    )
    assert deductible == pytest.approx(case.expected_deductible_yearly, abs=0.01)


@parametrized
def test_withholding_matches_the_payslip(case):
    contract = case.private_health
    deductible = private_deductible(
        contract.monthly_premium,
        contract.monthly_care_premium,
        contract.deductible_share,
        SYSTEM,
    )
    held = withholding(
        case.monthly_gross * 12.0,
        SYSTEM,
        tax_class=case.tax_class,
        n_children=case.n_children,
        private_deductible=deductible,
    )
    assert held.income_tax / 12.0 == pytest.approx(case.lohnsteuer, abs=0.02)
    assert held.solidarity / 12.0 == pytest.approx(case.soli, abs=0.01)


def household_for(case: PayslipCase) -> GermanHousehold:
    # Steuerklasse IV on the slip: the household combination is IV/IV, which
    # for a single listed earner degrades to plain class IV withholding.
    return GermanHousehold(
        earners=(
            Earner(monthly_gross=case.monthly_gross, private_health=case.private_health),
        ),
        joint=True,
        tax_classes="IV/IV",
        n_children=case.n_children,
    )


@parametrized
def test_household_net_matches_the_real_cash(case):
    from src.model.component import ModelParams, SetupContext

    state = household_for(case).open(
        SetupContext(
            ModelParams(
                years=1, start_age=None, yearly_inflation_rate=0.0, safe_withdrawal_rate=0.03
            )
        )
    )
    assert state.monthly_gross_total == pytest.approx(case.monthly_gross, abs=1e-9)
    # What is actually left each month: the payout on the slip minus what the
    # insurer debits from the account.
    assert state.monthly_net == pytest.approx(
        case.auszahlung - case.insurer_debit, abs=0.02
    )


@parametrized
def test_rebate_changes_cash_but_never_taxes(case):
    from src.model.component import ModelParams, SetupContext

    ctx = SetupContext(
        ModelParams(years=1, start_age=None, yearly_inflation_rate=0.0, safe_withdrawal_rate=0.03)
    )
    with_rebate = household_for(case).open(ctx)
    without = household_for(
        replace(case, private_health=replace(case.private_health, monthly_rebate=0.0))
    ).open(ctx)
    rebate = case.private_health.monthly_rebate
    assert with_rebate.monthly_net - without.monthly_net == pytest.approx(rebate, abs=1e-9)
    assert with_rebate.yearly_withheld == without.yearly_withheld
    assert with_rebate.yearly_assessed == without.yearly_assessed
