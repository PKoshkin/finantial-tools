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
entitlement assumed while employed (the full 14.6% health rate, and the
matching 4% haircut on the deduction), every adult is 23 or older (so the
childless care surcharge always applies to a household without children), and
the private insurance eligibility threshold (JAEG) is not checked.
Unemployment insurance is never deducted at assessment: the 1,900 EUR cap for
"sonstige Vorsorgeaufwendungen" is already exhausted by any full-time health
contribution, so the deduction would be zero anyway.

Retirement has its own three rules here: what a member of the
Krankenversicherung der Rentner pays out of a statutory pension
(`retiree_contributions`), what the pension fund pays toward a private premium
instead (`pension_subsidy`), and how much of a pension is taxable
(`besteuerungsanteil`). A pension owes no Krankengeld haircut: nothing it
pays for can turn into Krankengeld.

What an assessment leaves unpaid after the tax withheld at source — always
the case for a pension, which nothing is withheld from — the Finanzamt asks
for in advance the next year: `quarterly_prepayment` is the §37 EStG rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

BASE_YEAR = 2026  # the year TaxSystem's defaults are enacted for: model year 1
RENTENWERT = 40.79  # aktueller Rentenwert from 1 July 2025, in force at the model start


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
    bbg_knappschaft: float = 124_800.0  # knappschaftliche RV, yearly; sets the §10 Abs. 3 cap

    # --- assessment deductions -------------------------------------------
    employee_allowance: float = 1_230.0  # Arbeitnehmer-Pauschbetrag, per earner
    special_expenses_allowance: float = 36.0  # Sonderausgaben-Pauschbetrag, doubled jointly
    krankengeld_reduction: float = 0.04  # deductible health contribution is 96%
    other_vorsorge_cap: float = 1_900.0  # §10 Abs. 4 bucket; fixed by law, does not drift
    knappschaft_rv_rate: float = 0.247  # its contribution rate; the cap is this times its BBG
    pension_allowance: float = 102.0  # Werbungskosten-Pauschbetrag on pensions; does not drift

    # --- Solidaritätszuschlag ---------------------------------------------
    soli_rate: float = 0.055
    soli_allowance: float = 20_350.0  # Freigrenze on the tax, doubled jointly
    soli_phase_rate: float = 0.119  # Milderungszone cap on the excess

    # --- children ----------------------------------------------------------
    child_benefit_monthly: float = 259.0  # Kindergeld, per child
    child_allowance: float = 9_756.0  # Kinder- plus BEA-Freibetrag, per child

    # --- Vorauszahlungen (§37 Abs. 5 EStG); fixed by law, do not drift -------
    prepayment_minimum: float = 400.0  # a year's Vorauszahlungen, or none are set
    prepayment_raise_minimum: float = 100.0  # a raise per date, or the rate stands

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

        With the 45% border frozen, the 42% border drifts past it 56 years
        on and the tariff stops being one; that raises, and a horizon that
        long — a whole life, retirement included — needs
        `rich_threshold_frozen=False`.
        """
        if years_from_base == 0:
            return self
        t = (1.0 + self.yearly_threshold_growth) ** years_from_base
        w = (1.0 + self.yearly_wage_growth) ** years_from_base
        if self.rich_threshold_frozen and self.zone3_end * t >= self.rich_threshold:
            raise ValueError(
                f"{years_from_base} years on, the drifting 42% border passes the 45% one "
                f"frozen at {self.rich_threshold:,.0f}; a horizon this long needs "
                "TaxSystem(rich_threshold_frozen=False)"
            )
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
            bbg_knappschaft=self.bbg_knappschaft * w,
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


def employer_contributions(
    yearly_gross: float, system: TaxSystem, *, private_health: bool
) -> SocialContributions:
    """The employer's half, on the same bases and each capped on its own.

    Identical to the employee's share except for the two person-specific
    adjustments to the care rate: the childless surcharge and the per-child
    discount are the employee's alone, so an employer pays `pv_rate / 2` for
    everyone. For a privately insured employee the health and care slots are
    zero and the employer pays `private_subsidy` instead, exactly as
    `social_contributions` leaves them zero on the employee side.

    Deliberately absent, and therefore missing from any employer cost built
    on this: the Umlagen U1 and U2, the Insolvenzgeldumlage, and the accident
    insurance (Berufsgenossenschaft) — together roughly one to two percent of
    the gross, employer-only, and none of it ever reaching the employee.
    """
    pension_base = min(yearly_gross, system.bbg_pension)
    health_base = 0.0 if private_health else min(yearly_gross, system.bbg_health)
    return SocialContributions(
        pension=pension_base * (system.rv_rate / 2.0),
        unemployment=pension_base * (system.av_rate / 2.0),
        health=health_base * ((system.kv_base_rate + system.kv_extra_rate) / 2.0),
        care=health_base * (system.pv_rate / 2.0),
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
    *,
    subsidy: float | None = None,
) -> float:
    """The yearly deductible part of a private premium (§10 Abs. 1 Nr. 3).

    Only the insurer-certified Basisabsicherung share of the health premium
    counts, the care premium counts in full, and the employer subsidy is
    offset against the favored part in full — the comfort remainder yields
    nothing because the 1,900 EUR cap is already exhausted.

    `subsidy` is the tax-free monthly subsidy to offset, when it is not the
    employer's: in retirement the pension fund's `pension_subsidy`.
    """
    if subsidy is None:
        subsidy = private_subsidy(monthly_premium, monthly_care_premium, system)
    deductible = (
        deductible_share * monthly_premium + monthly_care_premium - subsidy
    ) * 12.0
    return max(deductible, 0.0)


def retiree_contributions(
    monthly_pension: float, system: TaxSystem, *, n_children: int
) -> SocialContributions:
    """What a KVdR member pays out of a statutory pension, monthly.

    The pension fund carries the other half of the health contribution, so
    the retiree pays half the general rate plus half the Zusatzbeitrag — the
    general rate, as there is no Krankengeld to price out. The care
    contribution falls on the retiree alone: both halves plus any childless
    surcharge (`pv_rate / 2` on top of the employee rate). Both are capped at
    the health ceiling; a pension owes no pension or unemployment
    contribution. A Basisrente is no part of the base: private pensions are
    contribution-free in the KVdR.
    """
    base = min(monthly_pension, system.bbg_health / 12.0)
    return SocialContributions(
        pension=0.0,
        unemployment=0.0,
        health=base * ((system.kv_base_rate + system.kv_extra_rate) / 2.0),
        care=base * (system.pv_rate / 2.0 + _employee_care_rate(system, n_children)),
    )


def pension_subsidy(monthly_pension: float, monthly_premium: float, system: TaxSystem) -> float:
    """The pension fund's monthly subsidy to a private health premium (§106 SGB VI).

    Half the general rate plus half the average Zusatzbeitrag, applied to the
    Zahlbetrag of the pension — what the fund would carry in the KVdR — and
    capped at half the health premium. Unlike the employer's subsidy it has
    no contribution ceiling and nothing for the care premium. Tax-free, so
    it is offset against the deductible premium.
    """
    return min(
        monthly_pension * ((system.kv_base_rate + system.kv_extra_rate) / 2.0),
        0.5 * monthly_premium,
    )


def besteuerungsanteil(start_year: int) -> float:
    """The taxable share of a pension starting in `start_year` (§22 Nr. 1 S. 3 a aa).

    The cohort rule as the Wachstumschancengesetz left it: 50% for every
    start up to 2005, then two points more per cohort to 80% in 2020, one
    point to 82% in 2022, and half a point from 2023 (82.5%) — 84% for a 2026
    start, 100% from 2058. Below 100% the share fixes a Rentenfreibetrag in
    euros, from the first full year's pension and for life, so every later
    Rentenanpassung is taxed in full; this model only takes pensions taxed
    in full from the first euro.
    """
    if start_year <= 2005:
        per_mille = 500
    elif start_year <= 2020:
        per_mille = 500 + 20 * (start_year - 2005)
    elif start_year <= 2022:
        per_mille = 800 + 10 * (start_year - 2020)
    else:
        per_mille = min(825 + 5 * (start_year - 2023), 1_000)
    return per_mille / 1_000.0


@dataclass(frozen=True, slots=True)
class EarnerFigures:
    """One earner's yearly figures, as the assessment needs them.

    `private_deductible` is 0.0 for a statutorily insured earner — their
    health and care deduction comes from `social` instead.

    A year in retirement adds `pension`, the gross statutory pension paid,
    and, in the KVdR, the `retiree_health` and `retiree_care` contributions
    paid out of it; a privately insured retiree's premium stays in
    `private_deductible`. A year in which the earner retires carries both
    halves: the salary months in `yearly_gross` and `social`, the pension
    months here.
    """

    yearly_gross: float
    social: SocialContributions
    private_deductible: float = 0.0
    pension: float = 0.0
    retiree_health: float = 0.0
    retiree_care: float = 0.0


def basisrente_cap(system: TaxSystem, *, joint: bool) -> float:
    """The §10 Abs. 3 Höchstbetrag for Altersvorsorgeaufwendungen.

    The maximum contribution to the knappschaftliche Rentenversicherung,
    rounded up to a full euro as the statute says — 27,566 EUR in 2024 (the
    figure a 2024 Basisrente offer quotes), 30,826 EUR in 2026 — and doubled
    for a jointly assessed couple, whichever spouse pays.
    """
    cap = float(math.ceil(system.knappschaft_rv_rate * system.bbg_knappschaft))
    return cap * (2.0 if joint else 1.0)


def basisrente_deduction(
    earners: Sequence[EarnerFigures],
    system: TaxSystem,
    *,
    joint: bool,
    contributions: float,
) -> float:
    """The part of a year's Basisrente contributions the cap lets through.

    §10 Abs. 3 puts both halves of the statutory pension contribution and the
    Basisrente under one cap, then takes the tax-free employer half back out.
    The employee half is deducted in full anyway — both halves stay far below
    the cap, because 18.6% of the RV ceiling is less than 24.7% of the
    knappschaft one and the two ceilings drift together — so the Basisrente
    gets whatever room the statutory contributions leave. The employer half
    equals the employee half to the bit: `employer_contributions` computes
    the same expression.
    """
    statutory = 0.0
    for earner in earners:
        statutory += 2.0 * earner.social.pension
    room = basisrente_cap(system, joint=joint) - statutory
    return min(contributions, max(room, 0.0))


def taxable_income(
    earners: Sequence[EarnerFigures],
    system: TaxSystem,
    *,
    joint: bool,
    other_income: float = 0.0,
    pension_income: float = 0.0,
    basisrente_contributions: float = 0.0,
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

    `pension_income` is a year's taxable Leibrente (§22 Nr. 1 S. 3 a aa),
    less the 102 EUR Werbungskosten-Pauschbetrag. The statute grants that
    allowance per person; it is applied once here, which is exact while the
    pensions belong to one spouse. With no earners at all this is the zvE of
    a retired tax unit.

    An earner's own `pension` — the statutory pension, taxed in full, which
    `GermanHousehold` only allows from a 2058 start — takes that earner's
    own allowance. Their KVdR contributions are deducted in full: no 4%
    haircut, since a pension carries no Krankengeld.

    `basisrente_contributions` are deducted as far as `basisrente_deduction`
    lets them through the §10 Abs. 3 cap.

    The unemployment contribution counts as far as it still fits the
    1,900 EUR bucket (§10 Abs. 4) next to the earner's health-and-care
    deduction — full on a statutory wage, so only a small private premium
    leaves it room. Other bucket dwellers (Krankentagegeld and comfort
    premium shares, liability insurance) are not modelled: the unemployment
    contribution alone fills whatever room a real wage leaves.
    """
    zve = 0.0
    for earner in earners:
        zve += max(earner.yearly_gross - system.employee_allowance, 0.0)
    zve += other_income
    if pension_income > 0.0:
        zve += max(pension_income - system.pension_allowance, 0.0)
    for earner in earners:
        if earner.pension > 0.0:
            zve += max(earner.pension - system.pension_allowance, 0.0)
    for earner in earners:
        zve -= earner.social.pension
        zve -= earner.social.health * (1.0 - system.krankengeld_reduction)
        zve -= earner.social.care
        zve -= earner.private_deductible
        zve -= earner.retiree_health
        zve -= earner.retiree_care
        basis = (
            earner.social.health * (1.0 - system.krankengeld_reduction)
            + earner.social.care
            + earner.private_deductible
            + earner.retiree_health
            + earner.retiree_care
        )
        room = system.other_vorsorge_cap - basis
        if room > 0.0:
            zve -= min(earner.social.unemployment, room)
    if basisrente_contributions > 0.0:
        zve -= basisrente_deduction(
            earners, system, joint=joint, contributions=basisrente_contributions
        )
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


def quarterly_prepayment(shortfall: float, system: TaxSystem, *, in_force: float = 0.0) -> float:
    """The Vorauszahlung per date that a Bescheid sets (§37 EStG).

    Paid on four dates a year — 10 March, June, September and December — and
    measured on the last assessment: the tax it set less what was withheld at
    source (§37 Abs. 3 S. 2), `shortfall`, income tax and Soli together, as
    the Soli is levied on the Vorauszahlungen too. A refund sets none, nor
    does a shortfall under 400 EUR a year (Abs. 5 S. 1). A rate already
    `in_force` is raised only by at least 100 EUR a date (S. 2), and lowered
    whenever the assessment says so.
    """
    if shortfall < system.prepayment_minimum:
        return 0.0
    quarterly = shortfall / 4.0
    if in_force < quarterly < in_force + system.prepayment_raise_minimum:
        return in_force
    return quarterly


def _vorsorgepauschale(
    yearly_wage: float,
    system: TaxSystem,
    *,
    n_children: int,
    private_deductible: float | None,
) -> float:
    """The 2026 payroll stand-in for the assessment's insurance deductions.

    Pension share plus the health-and-care part: for statutory insurance the
    health share at the *reduced* rate (the payroll convention) plus the care
    share, for private insurance the certified deductible premium in place of
    both. The unemployment contribution, part of the pauschale since 2026,
    only counts as far as it still fits the 1,900 EUR bucket next to the
    health-and-care part — on a statutory full-time wage the bucket is
    already full and it adds nothing, which is why only a small *private*
    premium feels it (pinned to the cent against a real 2026 payslip). The
    Mindestvorsorgepauschale was abolished from 2026.
    """
    pension_part = system.rv_rate / 2.0 * min(yearly_wage, system.bbg_pension)
    if private_deductible is not None:
        health_care_part = private_deductible
        pauschale = pension_part + private_deductible
    else:
        health_base = min(yearly_wage, system.bbg_health)
        health_part = (system.kv_reduced_rate + system.kv_extra_rate) / 2.0 * health_base
        care_part = _employee_care_rate(system, n_children) * health_base
        health_care_part = health_part + care_part
        pauschale = pension_part + health_part + care_part
    room = system.other_vorsorge_cap - health_care_part
    if room > 0.0:
        pauschale += min(system.av_rate / 2.0 * min(yearly_wage, system.bbg_pension), room)
    return pauschale


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


@dataclass(frozen=True, slots=True)
class EmployeeYear:
    """One employee's calendar year, from employer cost down to net cash.

    The whole chain in one place, for questions that are about a single year
    rather than about a forecast — "what does this salary actually cost, and
    what share of it arrives on the card". `GermanHousehold` runs the same
    arithmetic inside the simulation; this is the standalone form, and the two
    are pinned against each other in the tests.

    Both tax figures are kept because they answer different questions. The
    `withheld` amounts decide the twelve monthly payouts, i.e. the payslip;
    the `assessed` amounts are what the year finally owes once the tax return
    settles, and their difference is `refund` — cash that in the simulation
    only arrives the following July.
    """

    yearly_gross: float
    employer_social: SocialContributions
    employer_subsidy: float  # private only: the tax-free premium subsidy, yearly
    employee_social: SocialContributions
    private_premium: float  # yearly contract premium, health plus care
    private_rebate: float  # yearly Pauschalerstattung netted off the debit
    withheld: TaxAmounts
    assessed: TaxAmounts
    child_benefit: float

    @property
    def employer_cost(self) -> float:
        """The Arbeitgeberbrutto: gross plus everything the employer adds."""
        return self.yearly_gross + self.employer_social.total + self.employer_subsidy

    @property
    def refund(self) -> float:
        """Over-withholding the assessment gives back; negative means owed."""
        return self.withheld.total - self.assessed.total

    @property
    def net_paid_out(self) -> float:
        """The twelve Auszahlungsbeträge, before the assessment settles."""
        return (
            self.yearly_gross
            - self.employee_social.total
            - self.withheld.total
            - self.private_premium
            + self.private_rebate
            + self.employer_subsidy
            + self.child_benefit
        )

    @property
    def net(self) -> float:
        """What the year is finally worth, refund included."""
        return self.net_paid_out + self.refund

    @property
    def take_home_share(self) -> float:
        """`net` as a share of `employer_cost`; 0.0 for an unpaid year."""
        cost = self.employer_cost
        return self.net / cost if cost > 0.0 else 0.0


def employee_year(
    yearly_gross: float,
    system: TaxSystem,
    *,
    tax_class: str = "I",
    n_children: int = 0,
    joint: bool = False,
    monthly_premium: float | None = None,
    monthly_care_premium: float = 0.0,
    deductible_share: float = 1.0,
    monthly_rebate: float = 0.0,
) -> EmployeeYear:
    """One employee's full year: employer cost, contributions, tax, net.

    Defaults describe the plainest case — single, childless, statutorily
    insured, Steuerklasse I — which is also the only one where `joint` and
    `tax_class` need no thought. A married single earner is `joint=True` with
    `tax_class="III"`; the class alone never changes what the year owes, only
    how much of it is withheld monthly instead of refunded.

    `monthly_premium` is the private/statutory switch, the same `None`-means-
    statutory convention `withholding` uses: pass it (with the care premium
    and the insurer-certified `deductible_share`) to model a private contract,
    leave it out for GKV. `monthly_rebate` is a Pauschalerstattung, cash-only.

    Kindergeld is counted into the net, as `GermanHousehold` does, so a year
    with children is comparable to one without.
    """
    private = monthly_premium is not None
    employee_social = social_contributions(
        yearly_gross, system, n_children=n_children, private_health=private
    )
    employer_social = employer_contributions(yearly_gross, system, private_health=private)
    if private:
        deductible = private_deductible(
            monthly_premium, monthly_care_premium, deductible_share, system
        )
        subsidy = private_subsidy(monthly_premium, monthly_care_premium, system) * 12.0
        premium = (monthly_premium + monthly_care_premium) * 12.0
        rebate = monthly_rebate * 12.0
    else:
        deductible = None
        subsidy = 0.0
        premium = 0.0
        rebate = 0.0
    figures = EarnerFigures(
        yearly_gross=yearly_gross,
        social=employee_social,
        private_deductible=0.0 if deductible is None else deductible,
    )
    return EmployeeYear(
        yearly_gross=yearly_gross,
        employer_social=employer_social,
        employer_subsidy=subsidy,
        employee_social=employee_social,
        private_premium=premium,
        private_rebate=rebate,
        withheld=withholding(
            yearly_gross,
            system,
            tax_class=tax_class,
            n_children=n_children,
            private_deductible=deductible,
        ),
        assessed=assessed_tax(
            taxable_income([figures], system, joint=joint),
            system,
            joint=joint,
            n_children=n_children,
        ),
        child_benefit=n_children * 12.0 * system.child_benefit_monthly,
    )
