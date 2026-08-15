"""German employee taxation for one calendar year, base year 2026.

Pure functions over plain numbers — nothing in here knows about the simulation
engine. `TaxSystem` freezes the enacted 2026 figures as defaults, and
`TaxSystem.for_year()` extrapolates them by inertia for later years.

The statutory euro-rounding (§32a EStG: income floored to a full euro, the
tariff result floored, the splitting tax doubled after flooring the tax on
the half) is implemented faithfully: it is what makes these functions
comparable to the official BMF calculator and to published worked examples,
euro for euro. Cent fractions of the Solidaritätszuschlag are dropped, as the
statute says.

Scope cuts, all deliberate: no Kirchensteuer, no Steuerklasse II
(single-parent relief), no Sachsen Pflegeversicherung split, Krankengeld
entitlement assumed (the full 14.6% health rate, and the matching 4% haircut
on the deduction), every adult is 23 or older (so the childless care
surcharge always applies to a household without children), and the private
insurance eligibility threshold (JAEG) is not checked. Unemployment insurance
is never deducted at assessment: the 1,900 EUR cap for "sonstige
Vorsorgeaufwendungen" is already exhausted by any full-time health
contribution, so the deduction would be zero anyway.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence


@dataclass(frozen=True, slots=True)
class TaxSystem:
    """Every parameter of the German 2026 tax year, plus its inertia.

    The tariff is stored as its corner points — the zone borders and the
    marginal rates at them — rather than as the published polynomial
    coefficients, because the corner points are what the legislator actually
    moves each year. `tariff_zones` derives the coefficients back (and is
    pinned against the statutory 2026 figures in tests), so shifting the
    borders produces a valid future tariff automatically.

    Two inertia rates, because that is how the parameters actually move:
    tariff borders and allowances are shifted by law roughly with consumer
    prices (`yearly_threshold_growth`), while the contribution ceilings
    follow lagged wage growth by decree (`yearly_wage_growth`). Contribution
    *rates* are kept flat. The 45% border has been frozen at 277,826 EUR
    since 2022 as an explicit political choice, hence its own flag.
    """

    # --- §32a tariff corner points ---------------------------------------
    basic_allowance: float = 12_348.0  # Grundfreibetrag
    zone2_end: float = 17_799.0  # last euro of the first progression zone
    zone3_end: float = 69_878.0  # last euro of the second; 42% from the next
    rich_threshold: float = 277_826.0  # first euro taxed at 45%
    entry_rate: float = 0.14
    mid_rate: float = 0.2397
    top_rate: float = 0.42
    rich_rate: float = 0.45

    # --- social insurance (total rates; the employee pays half) ----------
    rv_rate: float = 0.186  # Rentenversicherung
    av_rate: float = 0.026  # Arbeitslosenversicherung
    kv_base_rate: float = 0.146  # Krankenversicherung, allgemeiner Satz
    kv_reduced_rate: float = 0.140  # ermäßigter Satz, used in the payroll pauschale
    kv_extra_rate: float = 0.029  # kassenindividueller Zusatzbeitrag (2026 average)
    pv_rate: float = 0.036  # Pflegeversicherung
    childless_surcharge: float = 0.006  # employee-only, from age 23
    per_child_discount: float = 0.0025  # employee-only, children #2..#5
    bbg_pension: float = 101_400.0  # Beitragsbemessungsgrenze RV/AV, yearly
    bbg_health: float = 69_750.0  # Beitragsbemessungsgrenze KV/PV, yearly

    # --- assessment deductions -------------------------------------------
    employee_allowance: float = 1_230.0  # Arbeitnehmer-Pauschbetrag, per earner
    special_expenses_allowance: float = 36.0  # Sonderausgaben-Pauschbetrag, doubled jointly
    krankengeld_reduction: float = 0.04  # deductible health contribution is 96%

    # --- Solidaritätszuschlag ---------------------------------------------
    soli_rate: float = 0.055
    soli_allowance: float = 20_350.0  # Freigrenze on the tax, doubled jointly
    soli_phase_rate: float = 0.119  # Milderungszone cap on the excess

    # --- children ----------------------------------------------------------
    child_benefit_monthly: float = 259.0  # Kindergeld, per child
    child_allowance: float = 9_756.0  # Kinder- plus BEA-Freibetrag, per child

    # --- Steuerklasse V withholding tiers (§39b Abs. 2 Satz 7) -------------
    class5_thresholds: tuple[float, float, float] = (14_071.0, 34_939.0, 222_260.0)

    # --- inertia ------------------------------------------------------------
    yearly_threshold_growth: float = 0.025
    yearly_wage_growth: float = 0.033
    rich_threshold_frozen: bool = True

    def __post_init__(self) -> None:
        if not (
            0.0 < self.basic_allowance < self.zone2_end < self.zone3_end < self.rich_threshold
        ):
            raise ValueError("tariff corner points must be increasing and positive")
        if not 0.0 < self.entry_rate < self.mid_rate < self.top_rate < self.rich_rate < 1.0:
            raise ValueError("tariff corner rates must be increasing and within (0, 1)")

    def for_year(self, years_from_base: int) -> TaxSystem:
        """The tax system `years_from_base` years after the base year.

        Always call this on the *base* system with an absolute offset — the
        result has the same inertia rates, so chaining `for_year` calls would
        compound the drift twice. Drifted borders are deliberately not
        re-rounded to full euros: the statutory flooring of income and tax is
        what carries the integer character, not the borders themselves.
        """
        if years_from_base == 0:
            return self
        t = (1.0 + self.yearly_threshold_growth) ** years_from_base
        w = (1.0 + self.yearly_wage_growth) ** years_from_base
        return replace(
            self,
            basic_allowance=self.basic_allowance * t,
            zone2_end=self.zone2_end * t,
            zone3_end=self.zone3_end * t,
            rich_threshold=(
                self.rich_threshold if self.rich_threshold_frozen else self.rich_threshold * t
            ),
            bbg_pension=self.bbg_pension * w,
            bbg_health=self.bbg_health * w,
            employee_allowance=self.employee_allowance * t,
            special_expenses_allowance=self.special_expenses_allowance * t,
            soli_allowance=self.soli_allowance * t,
            child_benefit_monthly=self.child_benefit_monthly * t,
            child_allowance=self.child_allowance * t,
            class5_thresholds=(
                self.class5_thresholds[0] * t,
                self.class5_thresholds[1] * t,
                self.class5_thresholds[2] * t,
            ),
        )


@dataclass(frozen=True, slots=True)
class TariffZones:
    """The §32a polynomial, derived from the tariff's corner points."""

    progression1: tuple[float, float]  # (a, b) of (a*y + b)*y
    progression2: tuple[float, float, float]  # (a, b, c) of (a*z + b)*z + c
    proportional_offset: float  # tax = top_rate*x - offset
    rich_offset: float  # tax = rich_rate*x - offset


def tariff_zones(system: TaxSystem) -> TariffZones:
    """Derive the §32a coefficients from the corner points.

    Inside a progression zone the marginal rate climbs linearly, so the tax
    is the integral of a trapezoid: the quadratic coefficient is half the
    slope (in the statute's per-10,000 units) and each zone constant is the
    unrounded cumulative tax at the zone border. Rounding the published
    figures to two decimals — the legislator's own convention — reproduces
    the statutory 2026 coefficients exactly (pinned in tests), which is what
    lets a drifted tariff be built from shifted borders instead of
    hand-copied coefficients.
    """
    width1 = system.zone2_end - system.basic_allowance
    width2 = system.zone3_end - system.zone2_end
    a1 = round((system.mid_rate - system.entry_rate) * 1e8 / (2.0 * width1), 2)
    b1 = round(system.entry_rate * 10_000.0, 2)
    a2 = round((system.top_rate - system.mid_rate) * 1e8 / (2.0 * width2), 2)
    b2 = round(system.mid_rate * 10_000.0, 2)
    cumulative1 = (system.entry_rate + system.mid_rate) / 2.0 * width1
    cumulative2 = cumulative1 + (system.mid_rate + system.top_rate) / 2.0 * width2
    proportional_end = system.rich_threshold - 1.0
    cumulative3 = cumulative2 + system.top_rate * (proportional_end - system.zone3_end)
    return TariffZones(
        progression1=(a1, b1),
        progression2=(a2, b2, round(cumulative1, 2)),
        proportional_offset=round(system.top_rate * system.zone3_end - cumulative2, 2),
        rich_offset=round(system.rich_rate * proportional_end - cumulative3, 2),
    )


def _tariff(x: float, system: TaxSystem) -> float:
    """§32a Abs. 1 on an already-floored (or exactly-half) income, unfloored.

    Takes the raw value on purpose: the splitting procedure evaluates the
    polynomial on the exact half of the floored joint income, which may end
    in .50 and is not floored again.
    """
    zones = tariff_zones(system)
    if x <= system.basic_allowance:
        return 0.0
    if x <= system.zone2_end:
        y = (x - system.basic_allowance) / 10_000.0
        a, b = zones.progression1
        return (a * y + b) * y
    if x <= system.zone3_end:
        z = (x - system.zone2_end) / 10_000.0
        a, b, c = zones.progression2
        return (a * z + b) * z + c
    if x < system.rich_threshold:
        return system.top_rate * x - zones.proportional_offset
    return system.rich_rate * x - zones.rich_offset


def income_tax(zve: float, system: TaxSystem) -> float:
    """The basic-tariff tax on a zu versteuerndes Einkommen, in full euros."""
    return float(math.floor(_tariff(float(math.floor(zve)), system)))


def splitting_tax(zve: float, system: TaxSystem) -> float:
    """The Ehegattensplitting tax on a joint zvE: twice the tax on the half.

    The joint income is floored, the half is not, and the flooring of the
    half's tax happens before doubling — so the result is always an even
    number of euros, exactly like the official computation.
    """
    return 2.0 * math.floor(_tariff(math.floor(zve) / 2.0, system))


def solidarity_surcharge(tax: float, system: TaxSystem, *, joint: bool) -> float:
    """Solidaritätszuschlag on an income-tax amount, cent fractions dropped.

    `tax` must already be the §51a-modified base: with children the child
    allowances are always deducted for this purpose, even when the
    Kindergeld won the Günstigerprüfung.
    """
    allowance = system.soli_allowance * (2.0 if joint else 1.0)
    if tax <= allowance:
        return 0.0
    surcharge = min(system.soli_rate * tax, system.soli_phase_rate * (tax - allowance))
    return math.floor(surcharge * 100.0) / 100.0


@dataclass(frozen=True, slots=True)
class SocialContributions:
    """One earner's employee-side contributions for a year."""

    pension: float
    unemployment: float
    health: float
    care: float

    @property
    def total(self) -> float:
        return self.pension + self.unemployment + self.health + self.care


def _employee_care_rate(system: TaxSystem, n_children: int) -> float:
    """The employee's Pflegeversicherung rate, children accounted for."""
    if n_children == 0:
        return system.pv_rate / 2.0 + system.childless_surcharge
    return system.pv_rate / 2.0 - system.per_child_discount * min(n_children - 1, 4)


def social_contributions(
    yearly_gross: float,
    system: TaxSystem,
    *,
    n_children: int,
    private_health: bool,
) -> SocialContributions:
    """Employee contributions on a yearly gross, each base capped on its own.

    A privately insured employee still owes pension and unemployment
    contributions; the health and care slots are zero because the premium is
    handled separately (see `private_subsidy` / `private_deductible`).
    """
    pension_base = min(yearly_gross, system.bbg_pension)
    health_base = min(yearly_gross, system.bbg_health)
    if private_health:
        health = 0.0
        care = 0.0
    else:
        health = health_base * ((system.kv_base_rate + system.kv_extra_rate) / 2.0)
        care = health_base * _employee_care_rate(system, n_children)
    return SocialContributions(
        pension=pension_base * (system.rv_rate / 2.0),
        unemployment=pension_base * (system.av_rate / 2.0),
        health=health,
        care=care,
    )


def private_subsidy(
    monthly_premium: float, monthly_care_premium: float, system: TaxSystem
) -> float:
    """The employer's tax-free monthly subsidy toward a private premium.

    Half of each premium, capped at half of the maximum statutory
    contribution — both caps are derived from the health ceiling, so they
    drift together with it (2026: 508.59 EUR health, 104.63 EUR care).
    """
    health_cap = 0.5 * (system.kv_base_rate + system.kv_extra_rate) * system.bbg_health / 12.0
    care_cap = system.pv_rate / 2.0 * system.bbg_health / 12.0
    return min(0.5 * monthly_premium, health_cap) + min(
        0.5 * monthly_care_premium, care_cap
    )


def private_deductible(
    monthly_premium: float,
    monthly_care_premium: float,
    deductible_share: float,
    system: TaxSystem,
) -> float:
    """The yearly deductible part of a private premium (§10 Abs. 1 Nr. 3).

    Only the insurer-certified Basisabsicherung share of the health premium
    counts, the care premium counts in full, and the employer subsidy is
    offset against the favored part in full — the comfort remainder yields
    nothing because the 1,900 EUR cap is already exhausted.
    """
    subsidy = private_subsidy(monthly_premium, monthly_care_premium, system)
    deductible = (
        deductible_share * monthly_premium + monthly_care_premium - subsidy
    ) * 12.0
    return max(deductible, 0.0)


@dataclass(frozen=True, slots=True)
class EarnerFigures:
    """One earner's yearly figures, as the assessment needs them.

    `private_deductible` is 0.0 for a statutorily insured earner — their
    health and care deduction comes from `social` instead.
    """

    yearly_gross: float
    social: SocialContributions
    private_deductible: float = 0.0


def taxable_income(
    earners: Sequence[EarnerFigures],
    system: TaxSystem,
    *,
    joint: bool,
    other_income: float = 0.0,
) -> float:
    """From gross wages to the zu versteuerndes Einkommen of one tax unit.

    Per earner: the Arbeitnehmer-Pauschbetrag (never below zero income),
    the full pension contribution (100% deductible since 2023), 96% of the
    health contribution (the Krankengeld haircut), the care contribution
    including any childless surcharge, and the deductible private premium.
    One Sonderausgaben-Pauschbetrag per spouse closes the chain.

    `other_income` is a yearly non-wage result — Einkünfte aus Vermietung
    und Verpachtung — joining the Summe der Einkünfte before the deductions.
    It may be negative: a rental loss offsets wage income in the same year,
    but whatever pushes the zvE below zero is forfeited — this model has no
    Verlustvortrag.
    """
    zve = 0.0
    for earner in earners:
        zve += max(earner.yearly_gross - system.employee_allowance, 0.0)
    zve += other_income
    for earner in earners:
        zve -= earner.social.pension
        zve -= earner.social.health * (1.0 - system.krankengeld_reduction)
        zve -= earner.social.care
        zve -= earner.private_deductible
    zve -= system.special_expenses_allowance * (2.0 if joint else 1.0)
    return max(zve, 0.0)


@dataclass(frozen=True, slots=True)
class TaxAmounts:
    """An income-tax amount and the surcharge riding on it."""

    income_tax: float
    solidarity: float

    @property
    def total(self) -> float:
        return self.income_tax + self.solidarity


def assessed_tax(
    zve: float, system: TaxSystem, *, joint: bool, n_children: int
) -> TaxAmounts:
    """The tax a Veranlagung settles on, Günstigerprüfung included.

    With children the child allowances are deducted only if the tax they
    save exceeds the Kindergeld already received — in which case the
    Kindergeld is added back to the tax (§31 EStG), so the assessed amount
    stays comparable to the withheld one. The Soli base always uses the
    child-reduced tax (§51a), whoever won.
    """
    tariff = splitting_tax if joint else income_tax
    base = tariff(zve, system)
    reduced = tariff(max(zve - n_children * system.child_allowance, 0.0), system)
    child_benefit = n_children * 12.0 * system.child_benefit_monthly
    if base - reduced > child_benefit:
        tax = reduced + child_benefit
    else:
        tax = base
    return TaxAmounts(
        income_tax=tax,
        solidarity=solidarity_surcharge(reduced, system, joint=joint),
    )


def _vorsorgepauschale(
    yearly_wage: float,
    system: TaxSystem,
    *,
    n_children: int,
    private_deductible: float | None,
) -> float:
    """The 2026 payroll stand-in for the assessment's insurance deductions.

    Pension share plus — for statutory insurance — the health share at the
    *reduced* rate (the payroll convention) and the care share; for private
    insurance, the certified deductible premium takes the place of both.
    The unemployment component introduced in 2026 is crowded out by its
    1,900 EUR cap at any full-time wage, so it is omitted entirely, and the
    Mindestvorsorgepauschale was abolished from 2026.
    """
    pension_part = system.rv_rate / 2.0 * min(yearly_wage, system.bbg_pension)
    if private_deductible is not None:
        return pension_part + private_deductible
    health_base = min(yearly_wage, system.bbg_health)
    health_part = (system.kv_reduced_rate + system.kv_extra_rate) / 2.0 * health_base
    care_part = _employee_care_rate(system, n_children) * health_base
    return pension_part + health_part + care_part


def _class_five_tax(base: float, system: TaxSystem) -> float:
    """Steuerklasse V per §39b Abs. 2 Satz 7: the doubled-difference formula.

    At least 14% of the base; the part above the first tier is capped at the
    top rate, which from the second tier onward *is* the marginal rate, and
    the rich rate takes over above the third.
    """
    first, second, third = system.class5_thresholds
    if base > third:
        return float(
            math.floor(system.top_rate * (third - first) + system.rich_rate * (base - third))
        )
    if base > second:
        return float(math.floor(system.top_rate * (base - first)))
    doubled = 2.0 * (income_tax(1.25 * base, system) - income_tax(0.75 * base, system))
    tax = max(doubled, math.floor(system.entry_rate * base))
    if base > first:
        tax = min(tax, system.top_rate * (base - first))
    return float(math.floor(tax))


def withholding(
    yearly_wage: float,
    system: TaxSystem,
    *,
    tax_class: str,
    n_children: int,
    private_deductible: float | None = None,
) -> TaxAmounts:
    """One earner's yearly Lohnsteuer and Soli under a Steuerklasse.

    The class never changes the assessed yearly tax — only how much of it is
    withheld at source, i.e. the cash timing until the refund. Class III
    applies the splitting tariff (the other spouse is presumed incomeless),
    IV and I the basic tariff, V the doubled-difference formula. For the
    Soli base, classes I/III/IV carry the child allowances (as the ELStAM
    do); class V carries none.
    """
    if tax_class not in ("I", "III", "IV", "V"):
        raise ValueError(f"unknown tax class: {tax_class!r}")
    pauschale = _vorsorgepauschale(
        yearly_wage, system, n_children=n_children, private_deductible=private_deductible
    )
    base = yearly_wage - system.employee_allowance - system.special_expenses_allowance
    base = max(base - pauschale, 0.0)
    if tax_class == "III":
        tax = splitting_tax(base, system)
    elif tax_class == "V":
        tax = _class_five_tax(base, system)
    else:
        tax = income_tax(base, system)
    if n_children > 0 and tax_class != "V":
        reduced_base = max(base - n_children * system.child_allowance, 0.0)
        if tax_class == "III":
            tax_for_soli = splitting_tax(reduced_base, system)
        else:
            tax_for_soli = income_tax(reduced_base, system)
    else:
        tax_for_soli = tax
    return TaxAmounts(
        income_tax=tax,
        solidarity=solidarity_surcharge(tax_for_soli, system, joint=tax_class == "III"),
    )
