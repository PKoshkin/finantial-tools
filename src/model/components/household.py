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
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    EarnerFigures,
    TaxSystem,
    assessed_tax,
    private_deductible,
    private_subsidy,
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
    """

    monthly_premium: float = 0.0
    monthly_care_premium: float = 60.0
    deductible_share: float = 0.9
    yearly_increase_rate: float = 0.035

    def __post_init__(self) -> None:
        check_amount(self.monthly_premium, "monthly_premium")
        check_amount(self.monthly_care_premium, "monthly_care_premium")
        if not 0.0 <= self.deductible_share <= 1.0:
            raise ValueError("deductible_share must be within [0, 1]")
        check_growth_rate(self.yearly_increase_rate, "yearly_increase_rate")


@dataclass(frozen=True)
class Earner:
    """One person's employment: a gross salary and how they are insured."""

    monthly_gross: float = 0.0
    yearly_increase_rate: float = 0.0
    private_health: PrivateHealthInsurance | None = None

    def __post_init__(self) -> None:
        check_amount(self.monthly_gross, "monthly_gross")
        check_growth_rate(self.yearly_increase_rate, "yearly_increase_rate")
        if self.private_health is not None and not isinstance(
            self.private_health, PrivateHealthInsurance
        ):
            raise TypeError("private_health must be a PrivateHealthInsurance or None")


@dataclass(slots=True)
class HouseholdState:
    monthly_gross: list[float]
    monthly_premiums: list[float]
    monthly_care_premiums: list[float]
    year_index: int = 0
    refund_due: float = 0.0  # last year's over-withholding, paid out in July
    monthly_net: float = 0.0
    monthly_gross_total: float = 0.0
    yearly_withheld: float = 0.0
    yearly_assessed: float = 0.0
    yearly_social: float = 0.0
    yearly_rental_result: float = 0.0
    figures: list[EarnerFigures] = field(default_factory=list)


@dataclass(frozen=True)
class GermanHousehold(FinancialComponent):
    """One or two gross salaries taxed together under German rules.

    A household is a single tax unit, which is why this is one SINGLETON
    component rather than two salary components: with the splitting tariff,
    the tax on two salaries is not the sum of two individually taxed ones.
    Reports the same `monthly_amount` key as `Salary`, so the flat
    `monthly_income` column keeps meaning "net cash per month" (the July
    refund is reported separately and excluded from it).

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
        )
        self._refresh(state)
        return state

    def _refresh(self, state: HouseholdState) -> None:
        """Recompute the year's payroll and assessment from current grosses.

        Everything a year pays is decided at its start — raises and premium
        increases only happen at year boundaries — so one computation per
        year is exact. Deliberately does not touch `refund_due`: that is
        last year's business.
        """
        system = self.tax.for_year(state.year_index)
        classes = self._classes()
        figures = []
        gross_total = 0.0
        net = 0.0
        withheld = 0.0
        social_total = 0.0
        for index, earner in enumerate(self.earners):
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
            figures.append(
                EarnerFigures(
                    yearly_gross=yearly_gross,
                    social=social,
                    private_deductible=0.0 if deductible is None else deductible,
                )
            )
            gross_total += state.monthly_gross[index]
            net += state.monthly_gross[index]
            net -= social.pension / 12.0
            net -= social.unemployment / 12.0
            net -= social.health / 12.0
            net -= social.care / 12.0
            net -= withheld_earner.income_tax / 12.0
            net -= withheld_earner.solidarity / 12.0
            if private:
                net -= state.monthly_premiums[index]
                net -= state.monthly_care_premiums[index]
                net += subsidy
            withheld += withheld_earner.income_tax
            withheld += withheld_earner.solidarity
            social_total += social.total
        net += self.n_children * system.child_benefit_monthly
        assessment = assessed_tax(
            taxable_income(figures, system, joint=self.joint),
            system,
            joint=self.joint,
            n_children=self.n_children,
        )
        state.monthly_gross_total = gross_total
        state.monthly_net = net
        state.yearly_withheld = withheld
        state.yearly_assessed = assessment.total
        state.yearly_social = social_total
        state.yearly_rental_result = 0.0
        state.figures = figures

    def year_begin(self, state: HouseholdState, ctx: YearContext) -> None:
        self._refresh(state)

    def cash_flows(self, state: HouseholdState, ctx: MonthContext) -> Sequence[CashFlow]:
        if ctx.month == 7:
            return (
                CashFlow(CashPhase.INCOME, state.monthly_net),
                CashFlow(CashPhase.INCOME, state.refund_due),
            )
        return (CashFlow(CashPhase.INCOME, state.monthly_net),)

    def year_assess(self, state: HouseholdState, ctx: YearContext) -> None:
        """Fold the year's rental results into the assessed tax.

        Runs at stage A2, after every property published its result in
        `year_settle` and before the row is reported — so the reported
        `yearly_assessed_tax` and the `refund_due` computed in `year_grow`
        both include the rental effect. `_refresh` next January resets the
        assessment to wage-only before the next round. When nothing was
        published the wage-only assessment stands untouched, bit for bit.
        """
        if not ctx.taxable_incomes:
            return
        rental = 0.0
        for amount in ctx.taxable_incomes:
            rental += amount
        state.yearly_rental_result = rental
        system = self.tax.for_year(state.year_index)
        assessment = assessed_tax(
            taxable_income(state.figures, system, joint=self.joint, other_income=rental),
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
        state.year_index += 1

    def report(self, state: HouseholdState, ctx: YearContext) -> dict[str, float]:
        return self._report(state)

    def report_initial(self, state: HouseholdState) -> dict[str, float]:
        return self._report(state)

    def _report(self, state: HouseholdState) -> dict[str, float]:
        return {
            "monthly_amount": state.monthly_net,
            "monthly_gross": state.monthly_gross_total,
            "yearly_assessed_tax": state.yearly_assessed,
            "yearly_withheld_tax": state.yearly_withheld,
            "yearly_refund_received": state.refund_due,
            "yearly_social_contributions": state.yearly_social,
            "yearly_rental_result": state.yearly_rental_result,
        }
