"""A German household: gross salaries in, net cash and a July refund out.

Unlike `Salary`, which takes the net figure as given, `GermanHousehold`
starts from one or two gross salaries and runs the German payroll and
assessment rules of `german_tax`: social contributions, wage-tax withholding
under a Steuerklasse choice, private or statutory health insurance, and the
joint yearly assessment whose over-withholding comes back as a refund.

The refund for a year is paid out on July 1 of the *following* year and
simply joins that month's income — the engine sweeps it into the sink like
any other leftover cash, so it starts compounding mid-year. Two
consequences: the very first simulated year contains no refund (there is no
prior year), and the final year's settlement arrives beyond the horizon and
is dropped. For wage over-withholding that cut is conservative — money owed
to the household disappears. With a taxed rental *profit* the dropped
settlement is a back-payment, so the last year's rental tax is never
charged and the final wealth is overstated by about one year's rental tax;
keep that in mind when comparing short horizons.

The Steuerklasse combination never changes the assessed yearly tax, only how
much cash arrives monthly versus as the July refund. "III/V" gives the first
earner class III; "IV/IV" withholds as if both were single, which for a
single-earner couple front-loads the tax and produces a large refund.

Rental components with a `RentalTaxation` publish their yearly Einkünfte aus
Vermietung und Verpachtung, and this household picks them up in `year_assess`:
the year's assessed tax is recomputed with the rental result in the zvE, so
the effect — a refund for a rental loss, a back-payment for a profit — rides
the same July channel as the wage over-withholding. Losses only offset income
within their year; whatever pushes the zvE below zero is forfeited (no
Verlustvortrag).

A `RuerupPension` publishes two things the same way. Its contributions are
deducted within the §10 Abs. 3 cap — at assessment only, since a
Basisrente is not part of the payroll Vorsorgepauschale and cannot be put
on the ELStAM as a Freibetrag — so the tax they save arrives as next July's
refund, exactly as it does in real life. Its pension joins the zvE as
taxable Leibrente; nothing is withheld at source, so the tax on it is
settled through the same channel, a year late (the model has no
Vorauszahlungen).

An `Earner` may retire, so that one run covers working life and old age:
after `Retirement.working_months` the salary stops and a Regelaltersrente
starts the next month. The pension is given in today's euros and follows
the Rentenwert every year from the model start, before Rentenbeginn as after
it. It is taxed in full, which is why the household only takes a
Rentenbeginn from 2058 on, when the Besteuerungsanteil reaches 100%: an
earlier cohort keeps a Rentenfreibetrag frozen in euros, which is not
modelled. Nothing is withheld from a pension, so a retired year's tax is a
back-payment with next July's settlement. A statutorily insured earner joins
the Krankenversicherung der Rentner: half the health contribution and the
whole care contribution come out of the pension and are deducted in full. A
privately insured one keeps the contract, and the pension fund pays half the
health premium, up to what it would carry in the KVdR (§106 SGB VI) — offset
against the deduction, as the employer's subsidy was. The year of retirement
holds both: its salary months are withheld and capped month by month, its
pension months follow, and one assessment covers the year. A Basisrente is
taxed as the first earner's: once that earner draws the statutory pension,
the two share one 102 EUR allowance, and neither the Basisrente nor any
other private pension owes KVdR contributions.

Deliberately out of scope in retirement: an early or deferred start (with
its Abschläge or Zuschläge), a pension that began before the model start,
Hinzuverdienst, Vorauszahlungen, survivors' pensions, a PKV premium falling
when its statutory surcharge stops at 65, and children growing up —
`n_children` keeps its Kindergeld and its care-rate effect for the whole
horizon.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

from ..component import (
    Cardinality,
    CashFlow,
    CashPhase,
    FinancialComponent,
    MonthContext,
    SetupContext,
    YearContext,
    check_amount,
    check_growth_rate,
)
from ..german_tax import (
    BASE_YEAR,
    RENTENWERT,
    EarnerFigures,
    SocialContributions,
    TaxSystem,
    assessed_tax,
    basisrente_deduction,
    besteuerungsanteil,
    pension_subsidy,
    private_deductible,
    private_subsidy,
    retiree_contributions,
    social_contributions,
    taxable_income,
    withholding,
)


@dataclass(frozen=True)
class PrivateHealthInsurance:
    """A private health contract, as payroll and assessment see it.

    The health premium and the mandatory care premium are separate because
    their employer subsidies are capped separately. `deductible_share` is
    the insurer-certified Basisabsicherung part of the health premium;
    `yearly_increase_rate` is the contract's own premium drift, applied on
    top of nothing else — private premiums move with claims inflation, not
    with the salary.

    `monthly_rebate` is a Pauschalerstattung the insurer nets against the
    debit (a Beitragsstundung): it reduces only the monthly cash outflow.
    Payroll and assessment keep using the full contract figures, exactly as
    they do in reality — and the rebate is contingent, lost for any year in
    which claims are filed, so modelling it is an optimistic choice. It
    escalates with the premiums.
    """

    monthly_premium: float = 0.0
    monthly_care_premium: float = 60.0
    deductible_share: float = 0.9
    yearly_increase_rate: float = 0.035
    monthly_rebate: float = 0.0

    def __post_init__(self) -> None:
        check_amount(self.monthly_premium, "monthly_premium")
        check_amount(self.monthly_care_premium, "monthly_care_premium")
        if not 0.0 <= self.deductible_share <= 1.0:
            raise ValueError("deductible_share must be within [0, 1]")
        check_growth_rate(self.yearly_increase_rate, "yearly_increase_rate")
        check_amount(self.monthly_rebate, "monthly_rebate")
        if self.monthly_rebate > self.monthly_premium + self.monthly_care_premium:
            raise ValueError("monthly_rebate must not exceed the premiums it discounts")


@dataclass(frozen=True)
class Retirement:
    """When one earner's salary stops, and the Regelaltersrente that follows.

    The salary is paid for `working_months` months from the model start and
    the pension from the next month on — the way a Regelaltersrente begins:
    the last salary for the month of the 67th birthday, the first pension on
    the first of the month after.

    `monthly_pension` is the gross pension in today's euros, at the
    Rentenwert in force at the model start — the figure a Renteninformation
    states. It follows the Rentenwert every year from the model start, before
    Rentenbeginn as after it, at `yearly_increase_rate`: None means the tax
    system's `yearly_wage_growth`, the wage growth the Rentenanpassung tracks
    by law. The real adjustment comes each July; like every yearly
    escalation in this model it is applied at the year boundary instead.
    """

    working_months: int = 0
    monthly_pension: float = 0.0
    yearly_increase_rate: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.working_months, int) or isinstance(self.working_months, bool):
            raise TypeError("working_months must be an int")
        if self.working_months < 0:
            raise ValueError("working_months must be >= 0")
        check_amount(self.monthly_pension, "monthly_pension")
        if self.yearly_increase_rate is not None:
            check_growth_rate(self.yearly_increase_rate, "yearly_increase_rate")

    @classmethod
    def from_entgeltpunkte(
        cls,
        entgeltpunkte: float,
        *,
        working_months: int,
        rentenwert: float = RENTENWERT,
        yearly_increase_rate: float | None = None,
    ) -> "Retirement":
        """The pension as Entgeltpunkte times the Rentenwert at the model start."""
        check_amount(entgeltpunkte, "entgeltpunkte")
        check_amount(rentenwert, "rentenwert")
        return cls(
            working_months=working_months,
            monthly_pension=entgeltpunkte * rentenwert,
            yearly_increase_rate=yearly_increase_rate,
        )

    @property
    def start_year(self) -> int:
        """The calendar year of the first pension, model year 1 being `BASE_YEAR`."""
        return BASE_YEAR + self.working_months // 12


@dataclass(frozen=True)
class Earner:
    """One person's employment: a gross salary, how they are insured, when it ends."""

    monthly_gross: float = 0.0
    yearly_increase_rate: float = 0.0
    private_health: PrivateHealthInsurance | None = None
    retirement: Retirement | None = None

    def __post_init__(self) -> None:
        check_amount(self.monthly_gross, "monthly_gross")
        check_growth_rate(self.yearly_increase_rate, "yearly_increase_rate")
        if self.private_health is not None and not isinstance(
            self.private_health, PrivateHealthInsurance
        ):
            raise TypeError("private_health must be a PrivateHealthInsurance or None")
        if self.retirement is not None and not isinstance(self.retirement, Retirement):
            raise TypeError("retirement must be a Retirement or None")


@dataclass(slots=True)
class HouseholdState:
    monthly_gross: list[float]
    monthly_premiums: list[float]
    monthly_care_premiums: list[float]
    monthly_rebates: list[float]
    monthly_pensions: list[float]  # each earner's, at this year's Rentenwert
    year_index: int = 0
    refund_due: float = 0.0  # last year's over-withholding, paid out in July
    monthly_net: float = 0.0  # the year's average; equal to every month's until a retirement
    monthly_nets: list[float] = field(default_factory=list)
    monthly_gross_total: float = 0.0
    monthly_pension_total: float = 0.0
    yearly_withheld: float = 0.0
    yearly_assessed: float = 0.0
    yearly_social: float = 0.0
    yearly_rental_result: float = 0.0
    yearly_basisrente_deduction: float = 0.0
    yearly_pension_income: float = 0.0
    figures: list[EarnerFigures] = field(default_factory=list)


@dataclass(frozen=True)
class GermanHousehold(FinancialComponent):
    """One or two gross salaries taxed together under German rules.

    A household is a single tax unit, which is why this is one SINGLETON
    component rather than two salary components: with the splitting tariff,
    the tax on two salaries is not the sum of two individually taxed ones.
    Reports the same `monthly_amount` key as `Salary`, so the flat
    `monthly_income` column keeps meaning "net cash per month" (the July
    refund is reported separately and excluded from it); in a year someone
    retires it is the average of the year's salary and pension months.
    `monthly_gross` and `monthly_pension` are the salaries and statutory
    pensions paid, averaged over the year the same way.

    `tax` holds the whole 2026 parameter set and its inertia; being a nested
    value object it is sweepable: `model.replaced("household.tax",
    kv_extra_rate=0.035)`.
    """

    default_name = "household"
    cardinality = Cardinality.SINGLETON
    report_kind = "salary"
    absorbs_taxable_income = True

    earners: tuple[Earner, ...] = ()
    joint: bool = True
    tax_classes: str = "III/V"
    n_children: int = 0
    tax: TaxSystem = TaxSystem()
    name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "earners", tuple(self.earners))
        if not 1 <= len(self.earners) <= 2:
            raise ValueError("earners must hold one or two Earner objects")
        for earner in self.earners:
            if not isinstance(earner, Earner):
                raise TypeError("earners must contain Earner objects")
        if not isinstance(self.joint, bool):
            raise TypeError("joint must be a bool")
        if len(self.earners) == 2 and not self.joint:
            raise ValueError("two earners are assessed jointly; set joint=True")
        if self.tax_classes not in ("III/V", "IV/IV"):
            raise ValueError('tax_classes must be "III/V" or "IV/IV"')
        if not isinstance(self.n_children, int) or isinstance(self.n_children, bool):
            raise TypeError("n_children must be an int")
        if self.n_children < 0:
            raise ValueError("n_children must be >= 0")
        if not isinstance(self.tax, TaxSystem):
            raise TypeError("tax must be a TaxSystem")
        for number, earner in enumerate(self.earners, start=1):
            if earner.retirement is None:
                continue
            start = earner.retirement.start_year
            share = besteuerungsanteil(start)
            if share < 1.0:
                raise ValueError(
                    f"earner {number}'s pension would start in {start}, taxable at "
                    f"{share:.1%} with a Rentenfreibetrag frozen in euros, which is not "
                    "modelled; the Rentenbeginn must be in 2058 or later "
                    f"(working_months >= {12 * (2058 - BASE_YEAR)})"
                )

    def _classes(self) -> tuple[str, ...]:
        """The Steuerklasse of each earner, first earner first."""
        if not self.joint:
            return ("I",)
        first, second = self.tax_classes.split("/")
        return (first, second)[: len(self.earners)]

    def open(self, ctx: SetupContext) -> HouseholdState:
        state = HouseholdState(
            monthly_gross=[earner.monthly_gross for earner in self.earners],
            monthly_premiums=[
                0.0 if earner.private_health is None else earner.private_health.monthly_premium
                for earner in self.earners
            ],
            monthly_care_premiums=[
                0.0
                if earner.private_health is None
                else earner.private_health.monthly_care_premium
                for earner in self.earners
            ],
            monthly_rebates=[
                0.0 if earner.private_health is None else earner.private_health.monthly_rebate
                for earner in self.earners
            ],
            monthly_pensions=[
                0.0 if earner.retirement is None else earner.retirement.monthly_pension
                for earner in self.earners
            ],
        )
        self._refresh(state)
        return state

    @staticmethod
    def _months_worked(earner: Earner, year_index: int) -> int:
        """How many months of the year `year_index` the earner draws a salary."""
        if earner.retirement is None:
            return 12
        return min(max(earner.retirement.working_months - 12 * year_index, 0), 12)

    def _refresh(self, state: HouseholdState) -> None:
        """Recompute the year's payroll, pensions and assessment.

        Everything a year pays is decided at its start — raises, premium and
        pension increases only happen at year boundaries, and a retirement
        falls on a month boundary known in advance — so one computation per
        year is exact. Each month's cash is summed earner by earner in one
        fixed order: a salary month as the payroll chain, a pension month as
        the pension less what is paid out of it. Until someone retires every
        month is the same chain, to the bit. Deliberately does not touch
        `refund_due`: that is last year's business.
        """
        system = self.tax.for_year(state.year_index)
        classes = self._classes()
        figures = []
        months_worked = []
        salary_terms = []
        pension_terms = []
        gross_total = 0.0
        pension_total = 0.0
        withheld = 0.0
        social_total = 0.0
        for index, earner in enumerate(self.earners):
            months = self._months_worked(earner, state.year_index)
            yearly_gross = state.monthly_gross[index] * 12.0
            private = earner.private_health is not None
            social = social_contributions(
                yearly_gross, system, n_children=self.n_children, private_health=private
            )
            if private:
                deductible = private_deductible(
                    state.monthly_premiums[index],
                    state.monthly_care_premiums[index],
                    earner.private_health.deductible_share,
                    system,
                )
                subsidy = private_subsidy(
                    state.monthly_premiums[index], state.monthly_care_premiums[index], system
                )
            else:
                deductible = None
                subsidy = 0.0
            withheld_earner = withholding(
                yearly_gross,
                system,
                tax_class=classes[index],
                n_children=self.n_children,
                private_deductible=deductible,
            )
            # A salary month, as signed terms in the order they are summed.
            salary = [
                state.monthly_gross[index],
                -(social.pension / 12.0),
                -(social.unemployment / 12.0),
                -(social.health / 12.0),
                -(social.care / 12.0),
                -(withheld_earner.income_tax / 12.0),
                -(withheld_earner.solidarity / 12.0),
            ]
            if private:
                salary += [
                    -state.monthly_premiums[index],
                    -state.monthly_care_premiums[index],
                    subsidy,
                    state.monthly_rebates[index],
                ]
            months_worked.append(months)
            salary_terms.append(salary)
            if months == 12:
                pension_terms.append([])
                figures.append(
                    EarnerFigures(
                        yearly_gross=yearly_gross,
                        social=social,
                        private_deductible=0.0 if deductible is None else deductible,
                    )
                )
                gross_total += state.monthly_gross[index]
                withheld += withheld_earner.income_tax
                withheld += withheld_earner.solidarity
                social_total += social.total
                continue

            # The pension months, and what the year's salary months add up to.
            retired = 12 - months
            pension = state.monthly_pensions[index]
            if private:
                fund_subsidy = pension_subsidy(pension, state.monthly_premiums[index], system)
                retired_deductible = private_deductible(
                    state.monthly_premiums[index],
                    state.monthly_care_premiums[index],
                    earner.private_health.deductible_share,
                    system,
                    subsidy=fund_subsidy,
                )
                retiree = SocialContributions(0.0, 0.0, 0.0, 0.0)
                pension_terms.append(
                    [
                        pension,
                        -state.monthly_premiums[index],
                        -state.monthly_care_premiums[index],
                        fund_subsidy,
                        state.monthly_rebates[index],
                    ]
                )
            else:
                retired_deductible = 0.0
                retiree = retiree_contributions(pension, system, n_children=self.n_children)
                pension_terms.append([pension, -retiree.health, -retiree.care])
            working_deductible = 0.0 if deductible is None else deductible
            figures.append(
                EarnerFigures(
                    yearly_gross=state.monthly_gross[index] * months,
                    social=SocialContributions(
                        pension=social.pension / 12.0 * months,
                        unemployment=social.unemployment / 12.0 * months,
                        health=social.health / 12.0 * months,
                        care=social.care / 12.0 * months,
                    ),
                    private_deductible=(
                        working_deductible / 12.0 * months + retired_deductible / 12.0 * retired
                    ),
                    pension=pension * retired,
                    retiree_health=retiree.health * retired,
                    retiree_care=retiree.care * retired,
                )
            )
            gross_total += state.monthly_gross[index] / 12.0 * months
            pension_total += pension / 12.0 * retired
            withheld += withheld_earner.income_tax / 12.0 * months
            withheld += withheld_earner.solidarity / 12.0 * months
            social_total += social.total / 12.0 * months
            social_total += (retiree.health + retiree.care) * retired
        nets = []
        for month in range(12):
            net = 0.0
            for months, salary, pension_month in zip(months_worked, salary_terms, pension_terms):
                for term in salary if month < months else pension_month:
                    net += term
            net += self.n_children * system.child_benefit_monthly
            nets.append(net)
        if all(net == nets[0] for net in nets):
            average = nets[0]
        else:
            average = 0.0
            for net in nets:
                average += net
            average /= 12.0
        assessment = assessed_tax(
            taxable_income(figures, system, joint=self.joint),
            system,
            joint=self.joint,
            n_children=self.n_children,
        )
        state.monthly_gross_total = gross_total
        state.monthly_pension_total = pension_total
        state.monthly_nets = nets
        state.monthly_net = average
        state.yearly_withheld = withheld
        state.yearly_assessed = assessment.total
        state.yearly_social = social_total
        state.yearly_rental_result = 0.0
        state.yearly_basisrente_deduction = 0.0
        state.yearly_pension_income = 0.0
        state.figures = figures

    def year_begin(self, state: HouseholdState, ctx: YearContext) -> None:
        self._refresh(state)

    def cash_flows(self, state: HouseholdState, ctx: MonthContext) -> Sequence[CashFlow]:
        net = state.monthly_nets[ctx.month - 1]
        if ctx.month == 7:
            return (
                CashFlow(CashPhase.INCOME, net),
                CashFlow(CashPhase.INCOME, state.refund_due),
            )
        return (CashFlow(CashPhase.INCOME, net),)

    def year_assess(self, state: HouseholdState, ctx: YearContext) -> None:
        """Fold the year's published results into the assessed tax.

        Runs at stage A2, after every component published in `year_settle`
        and before the row is reported — so the reported
        `yearly_assessed_tax` and the `refund_due` computed in `year_grow`
        both include the rental result, the Basisrente deduction and the
        pension income. `_refresh` next January resets the assessment to
        wages and statutory pensions only before the next round. When nothing
        was published that assessment stands untouched, bit for bit.

        A Basisrente is taxed as the first earner's pension: in a year that
        earner also draws the statutory one, the two are one person's
        pensions and share one 102 EUR allowance.
        """
        if not (ctx.taxable_incomes or ctx.basisrente_contributions or ctx.pension_incomes):
            return
        rental = 0.0
        for amount in ctx.taxable_incomes:
            rental += amount
        pension = 0.0
        for amount in ctx.pension_incomes:
            pension += amount
        contributions = 0.0
        for amount in ctx.basisrente_contributions:
            contributions += amount
        state.yearly_rental_result = rental
        state.yearly_pension_income = pension
        figures = state.figures
        leibrente = pension
        if pension > 0.0 and figures[0].pension > 0.0:
            figures = [replace(figures[0], pension=figures[0].pension + pension), *figures[1:]]
            leibrente = 0.0
        system = self.tax.for_year(state.year_index)
        if contributions > 0.0:
            state.yearly_basisrente_deduction = basisrente_deduction(
                figures, system, joint=self.joint, contributions=contributions
            )
        assessment = assessed_tax(
            taxable_income(
                figures,
                system,
                joint=self.joint,
                other_income=rental,
                pension_income=leibrente,
                basisrente_contributions=contributions,
            ),
            system,
            joint=self.joint,
            n_children=self.n_children,
        )
        state.yearly_assessed = assessment.total

    def year_grow(self, state: HouseholdState, ctx: YearContext) -> None:
        state.refund_due = state.yearly_withheld - state.yearly_assessed
        for index, earner in enumerate(self.earners):
            state.monthly_gross[index] *= 1.0 + earner.yearly_increase_rate
            if earner.private_health is not None:
                rate = earner.private_health.yearly_increase_rate
                state.monthly_premiums[index] *= 1.0 + rate
                state.monthly_care_premiums[index] *= 1.0 + rate
                state.monthly_rebates[index] *= 1.0 + rate
            if earner.retirement is not None:
                rate = earner.retirement.yearly_increase_rate
                if rate is None:
                    rate = self.tax.yearly_wage_growth
                state.monthly_pensions[index] *= 1.0 + rate
        state.year_index += 1

    def report(self, state: HouseholdState, ctx: YearContext) -> dict[str, float]:
        return self._report(state)

    def report_initial(self, state: HouseholdState) -> dict[str, float]:
        return self._report(state)

    def _report(self, state: HouseholdState) -> dict[str, float]:
        return {
            "monthly_amount": state.monthly_net,
            "monthly_gross": state.monthly_gross_total,
            "monthly_pension": state.monthly_pension_total,
            "yearly_assessed_tax": state.yearly_assessed,
            "yearly_withheld_tax": state.yearly_withheld,
            "yearly_refund_received": state.refund_due,
            "yearly_social_contributions": state.yearly_social,
            "yearly_rental_result": state.yearly_rental_result,
            "yearly_basisrente_deduction": state.yearly_basisrente_deduction,
            "yearly_pension_income": state.yearly_pension_income,
        }
