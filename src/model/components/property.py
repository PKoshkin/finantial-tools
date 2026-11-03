"""Real-estate components: your own home, buy-to-let flats, and the mortgage.

`Mortgage` is a nested value object rather than a component: a loan in this
model always belongs to exactly one property, and the down payment, the fees
and the loan are all derived from that property's price. `OperatingCosts` and
`RentalTaxation` follow the same pattern for a buy-to-let flat's
Bewirtschaftungskosten and its Vermietung-und-Verpachtung taxation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..component import (
    Cardinality,
    CashDraw,
    CashFlow,
    CashPhase,
    Compounding,
    FinancialComponent,
    MonthContext,
    SetupContext,
    YearContext,
    check_amount,
    check_compounding,
    check_growth_rate,
)


@dataclass(frozen=True)
class Mortgage:
    """A German annuity loan: fixed payment, interest plus Tilgung.

    The payment is recomputed from the outstanding principal every
    `refinancing_years` (the end of a Zinsbindung), and stops entirely once the
    loan is repaid.

    `compounding` says how `interest_rate` accrues month to month. A Sollzins
    is a nominal rate with monthly accrual, so the default is `NOMINAL`: 3.8%
    a year costs 3.8%/12 a month and works out to 3.867% effective, which is
    the loan the bank is actually selling. It also decides how the annuity is
    split, so that `yearly_repayment_rate` always means what it says: the
    payment is the month's interest plus `loan * yearly_repayment_rate / 12`
    of principal, which for the nominal default is the familiar
    `loan * (interest_rate + yearly_repayment_rate) / 12`.
    """

    interest_rate: float = 0.0
    down_payment_rate: float = 0.0
    fees_rate: float = 0.0  # notary, taxes, agent: % of price, sunk
    yearly_repayment_rate: float = 0.0  # initial Tilgung
    refinancing_years: int = 0  # 0 -> the annuity is never recomputed
    sondertilgung_yearly_rate: float = 0.0  # % of the ORIGINAL loan per year
    compounding: Compounding = Compounding.NOMINAL

    def __post_init__(self) -> None:
        if not isinstance(self.refinancing_years, int) or isinstance(
            self.refinancing_years, bool
        ):
            raise TypeError("refinancing_years must be an int")
        for name in (
            "interest_rate",
            "down_payment_rate",
            "fees_rate",
            "yearly_repayment_rate",
            "sondertilgung_yearly_rate",
        ):
            check_amount(getattr(self, name), name)
        if self.refinancing_years < 0:
            raise ValueError("refinancing_years must be >= 0")
        if not 0.0 <= self.down_payment_rate <= 1.0:
            raise ValueError("down_payment_rate must be within [0, 1]")
        check_compounding(self.compounding)

    def annuity_payment(self, loan: float) -> float:
        """Monthly payment servicing `loan` at this mortgage's rates.

        Interest plus the initial Tilgung. The nominal branch is written as the
        single expression the pinned figures were computed with — splitting it
        into two terms would change the last bits.
        """
        if loan <= 0:
            return 0.0
        if self.compounding is Compounding.NOMINAL:
            return loan * (self.interest_rate + self.yearly_repayment_rate) / 12.0
        return self.monthly_interest(loan) + loan * self.yearly_repayment_rate / 12.0

    def monthly_interest(self, loan: float) -> float:
        """Interest accrued on `loan` over one month."""
        return self.compounding.monthly_charge(loan, self.interest_rate)


@dataclass(frozen=True)
class OperatingCosts:
    """Bewirtschaftungskosten of a rental flat: the non-recoverable part.

    Both amounts are per month at t=0 and escalate together once a year by
    `yearly_increase_rate` (the Kostensteigerung). The split exists for the
    tax return: `monthly_deductible` — the non-recoverable Hausgeld, the
    Grundsteuer and other actual expenses — is a Werbungskosten deduction,
    while `monthly_own_reserve` (the eigene Instandhaltungsrücklage) is money
    set aside, a cash cost that deducts nothing. Costs the tenant reimburses
    (umlagefähige Kosten) are not modelled at all: they cancel between the
    Warmmiete and the expenses, so the rent stays a Nettokaltmiete.
    """

    monthly_deductible: float = 0.0
    monthly_own_reserve: float = 0.0
    yearly_increase_rate: float = 0.0

    def __post_init__(self) -> None:
        check_amount(self.monthly_deductible, "monthly_deductible")
        check_amount(self.monthly_own_reserve, "monthly_own_reserve")
        check_growth_rate(self.yearly_increase_rate, "yearly_increase_rate")


@dataclass(frozen=True)
class RentalTaxation:
    """How a buy-to-let flat enters the household's income tax.

    Setting this on an `InvestmentProperty` makes it publish its yearly
    Einkünfte aus Vermietung und Verpachtung — rent minus deductible costs,
    loan interest and AfA — to the model's `GermanHousehold`, whose assessed
    tax then moves accordingly (the cash arrives with the July settlement of
    the following year, and a profit's tax is then also asked for in
    advance). The AfA base is `(price + purchase fees) * building_share`
    and depreciates at `afa_rate` a year (§7 EStG: 2% for existing buildings,
    3% for new ones), flat forever — the model does not stop depreciating
    once the building is written off, which overstates the deduction beyond
    `1 / afa_rate` years.
    """

    afa_rate: float
    building_share: float

    def __post_init__(self) -> None:
        check_amount(self.afa_rate, "afa_rate")
        if self.afa_rate > 1.0:
            raise ValueError("afa_rate must be within [0, 1]")
        check_amount(self.building_share, "building_share")
        if self.building_share > 1.0:
            raise ValueError("building_share must be within [0, 1]")


@dataclass(slots=True)
class PropertyState:
    loan: float
    payment: float
    reserve_base: float
    down_payment: float
    purchase_fees: float
    monthly_rental_income: float = 0.0
    month_payment: float = 0.0
    month_reserve: float = 0.0
    year_interest: float = 0.0
    year_principal: float = 0.0
    year_mortgage_paid: float = 0.0
    sondertilgung_reserved: float = 0.0
    yearly_sondertilgung: float = 0.0
    cumulative_interest: float = 0.0
    monthly_deductible_costs: float = 0.0
    monthly_own_reserve: float = 0.0
    monthly_afa: float = 0.0
    year_taxable_result: float = 0.0
    cumulative_cashflow: float = 0.0


class _PropertyBase(FinancialComponent):
    """Shared behaviour of every kind of property.

    Concrete subclasses declare the dataclass fields (so their signatures stay
    readable) and supply the rental income, which only buy-to-let has.
    """

    report_kind = "property"

    # Declared by the subclasses.
    price: float
    yearly_price_increase_rate: float
    mortgage: Mortgage | None

    # --- validation shared by the subclasses ---
    def _validate(self) -> None:
        check_amount(self.price, "price")
        check_growth_rate(self.yearly_price_increase_rate, "yearly_price_increase_rate")
        if self.mortgage is not None and not isinstance(self.mortgage, Mortgage):
            raise TypeError("mortgage must be a Mortgage or None")

    # --- derived amounts, handy in notebooks ---
    @property
    def down_payment(self) -> float:
        return 0.0 if self.mortgage is None else self.price * self.mortgage.down_payment_rate

    @property
    def purchase_fees(self) -> float:
        return 0.0 if self.mortgage is None else self.price * self.mortgage.fees_rate

    @property
    def initial_loan(self) -> float:
        return 0.0 if self.mortgage is None else self.price - self.down_payment

    @property
    def initial_monthly_payment(self) -> float:
        return 0.0 if self.mortgage is None else self.mortgage.annuity_payment(self.initial_loan)

    @property
    def max_yearly_sondertilgung(self) -> float:
        if self.mortgage is None:
            return 0.0
        return self.initial_loan * self.mortgage.sondertilgung_yearly_rate

    # --- rental income: overridden by InvestmentProperty ---
    def _initial_rental_income(self) -> float:
        return 0.0

    def _rental_increase_rate(self) -> float:
        return 0.0

    # --- t = 0 ---
    def open(self, ctx: SetupContext) -> PropertyState:
        if self.mortgage is None:
            return PropertyState(
                loan=0.0,
                payment=0.0,
                reserve_base=0.0,
                down_payment=0.0,
                purchase_fees=0.0,
                monthly_rental_income=self._initial_rental_income(),
            )
        down_payment = self.price * self.mortgage.down_payment_rate
        purchase_fees = self.price * self.mortgage.fees_rate
        loan = self.price - down_payment
        return PropertyState(
            loan=loan,
            payment=self.mortgage.annuity_payment(loan),
            reserve_base=(loan * self.mortgage.sondertilgung_yearly_rate) / 12.0,
            down_payment=down_payment,
            purchase_fees=purchase_fees,
            monthly_rental_income=self._initial_rental_income(),
        )

    def initial_draws(self, state: PropertyState) -> Sequence[CashDraw]:
        if self.mortgage is None:
            return (CashDraw(self.price, "purchase"),)
        return (
            CashDraw(state.down_payment, "down_payment"),
            CashDraw(state.purchase_fees, "fees"),
        )

    # --- the year ---
    def year_begin(self, state: PropertyState, ctx: YearContext) -> None:
        state.year_interest = 0.0
        state.year_principal = 0.0
        state.year_mortgage_paid = 0.0
        state.sondertilgung_reserved = 0.0
        state.yearly_sondertilgung = 0.0

    def month_begin(self, state: PropertyState, ctx: MonthContext) -> None:
        if self.mortgage is not None and state.loan > 0:
            interest = self.mortgage.monthly_interest(state.loan)
            principal = min(state.payment - interest, state.loan)
            payment = interest + principal
            state.loan -= principal
            if state.loan <= 0:
                state.loan = 0.0
                state.payment = 0.0
            # The reserve is charged for every month the loan was still alive
            # at the start of it — including the month it gets repaid in.
            state.month_reserve = state.reserve_base
        else:
            interest = 0.0
            principal = 0.0
            payment = 0.0
            state.month_reserve = 0.0

        state.month_payment = payment
        state.year_interest += interest
        state.year_principal += principal
        state.year_mortgage_paid += payment
        state.sondertilgung_reserved += state.month_reserve

    def cash_flows(self, state: PropertyState, ctx: MonthContext) -> Sequence[CashFlow]:
        return (CashFlow(CashPhase.DEBT_SERVICE, -state.month_payment),)

    def claim(self, state: PropertyState, ctx: MonthContext) -> float:
        return state.month_reserve

    # --- year end ---
    def year_settle(self, state: PropertyState, ctx: YearContext) -> None:
        applied = min(state.sondertilgung_reserved, state.loan)
        if applied > 0:
            state.loan -= applied
            if state.loan <= 0:
                state.loan = 0.0
                state.payment = 0.0
        unused = state.sondertilgung_reserved - applied
        if unused > 0:
            # The loan cannot swallow the reserve any more; the cash goes back
            # to the sink instead of evaporating.
            ctx.release(unused)
        state.yearly_sondertilgung = applied
        state.cumulative_interest += state.year_interest

    def year_grow(self, state: PropertyState, ctx: YearContext) -> None:
        state.monthly_rental_income *= 1.0 + self._rental_increase_rate()

    def year_refinance(self, state: PropertyState, ctx: YearContext) -> None:
        if self.mortgage is None or state.loan <= 0:
            return
        if self.mortgage.refinancing_years > 0 and ctx.year % self.mortgage.refinancing_years == 0:
            state.payment = self.mortgage.annuity_payment(state.loan)

    # --- reporting ---
    def report(self, state: PropertyState, ctx: YearContext) -> dict[str, float]:
        property_value = self.price * (1.0 + self.yearly_price_increase_rate) ** ctx.year
        return {
            "total_loan": state.loan,
            "property_value": property_value,
            "property_equity": property_value - state.loan,
            "monthly_rental_income": state.monthly_rental_income,
            "monthly_interest_payment": state.year_interest / 12.0,
            "monthly_loan_repayment": state.year_principal / 12.0,
            "monthly_mortgage_payment": state.year_mortgage_paid / 12.0,
            "yearly_sondertilgung": state.yearly_sondertilgung,
            "cumulative_interest_paid": state.cumulative_interest,
        }

    def report_initial(self, state: PropertyState) -> dict[str, float]:
        return {
            "total_loan": state.loan,
            "property_value": self.price,
            "property_equity": self.price - state.loan,
            "monthly_rental_income": state.monthly_rental_income,
            "monthly_interest_payment": 0.0,
            "monthly_loan_repayment": 0.0,
            "monthly_mortgage_payment": state.payment,
            "yearly_sondertilgung": 0.0,
            "cumulative_interest_paid": 0.0,
        }


@dataclass(frozen=True)
class Home(_PropertyBase):
    """The flat or house you live in. At most one per model.

    `mortgage=None` means an all-cash purchase with no transaction costs; for a
    cash purchase that still pays notary and taxes use
    `Mortgage(down_payment_rate=1.0, fees_rate=...)`.
    """

    default_name = "home"
    cardinality = Cardinality.SINGLETON

    price: float = 0.0
    yearly_price_increase_rate: float = 0.0
    mortgage: Mortgage | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        self._validate()


@dataclass(frozen=True)
class InvestmentProperty(_PropertyBase):
    """A flat bought to rent out. Any number of them per model.

    By default the rental income is plain tax-free cash, exactly as before
    the refactoring. `operating_costs` adds the non-recoverable
    Bewirtschaftungskosten, `vacancy_rate` a kalkulatorischer Mietausfall,
    and `taxation` sends the flat's Einkünfte aus Vermietung und Verpachtung
    into the model's `GermanHousehold`. A vacancy reduces the collected rent
    and with it the taxable result — unlike the planning calculators that
    book the Mietausfall as a non-deductible reserve on top of full rent.
    """

    default_name = "flat"
    cardinality = Cardinality.MULTI

    price: float = 0.0
    yearly_price_increase_rate: float = 0.0
    monthly_rental_income: float = 0.0
    yearly_rental_income_increase_rate: float = 0.0
    mortgage: Mortgage | None = None
    operating_costs: OperatingCosts | None = None
    vacancy_rate: float = 0.0
    taxation: RentalTaxation | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        self._validate()
        check_amount(self.monthly_rental_income, "monthly_rental_income")
        check_growth_rate(
            self.yearly_rental_income_increase_rate, "yearly_rental_income_increase_rate"
        )
        if self.operating_costs is not None and not isinstance(
            self.operating_costs, OperatingCosts
        ):
            raise TypeError("operating_costs must be an OperatingCosts or None")
        check_amount(self.vacancy_rate, "vacancy_rate")
        if self.vacancy_rate > 1.0:
            raise ValueError("vacancy_rate must be within [0, 1]")
        if self.taxation is not None and not isinstance(self.taxation, RentalTaxation):
            raise TypeError("taxation must be a RentalTaxation or None")

    @property
    def produces_taxable_income(self) -> bool:
        return self.taxation is not None

    # --- derived amounts, handy in notebooks ---
    @property
    def afa_base(self) -> float:
        """The depreciable acquisition cost: building share of price plus fees."""
        if self.taxation is None:
            return 0.0
        return (self.price + self.purchase_fees) * self.taxation.building_share

    @property
    def monthly_afa(self) -> float:
        return 0.0 if self.taxation is None else self.afa_base * self.taxation.afa_rate / 12.0

    def _initial_rental_income(self) -> float:
        return self.monthly_rental_income

    def _rental_increase_rate(self) -> float:
        return self.yearly_rental_income_increase_rate

    def _effective_rent(self, state: PropertyState) -> float:
        # `x * 1.0 == x` exactly in IEEE 754, so a zero vacancy is bit-safe.
        return state.monthly_rental_income * (1.0 - self.vacancy_rate)

    def open(self, ctx: SetupContext) -> PropertyState:
        state = super().open(ctx)
        if self.operating_costs is not None:
            state.monthly_deductible_costs = self.operating_costs.monthly_deductible
            state.monthly_own_reserve = self.operating_costs.monthly_own_reserve
        state.monthly_afa = self.monthly_afa
        if self.mortgage is None:
            state.cumulative_cashflow = -self.price
        else:
            state.cumulative_cashflow = -(state.down_payment + state.purchase_fees)
        return state

    def cash_flows(self, state: PropertyState, ctx: MonthContext) -> Sequence[CashFlow]:
        flows = [
            CashFlow(CashPhase.RENTAL_INCOME, self._effective_rent(state)),
            CashFlow(CashPhase.DEBT_SERVICE, -state.month_payment),
        ]
        if self.operating_costs is not None:
            flows.insert(
                1,
                CashFlow(
                    CashPhase.OPERATING_COSTS,
                    -(state.monthly_deductible_costs + state.monthly_own_reserve),
                ),
            )
        return tuple(flows)

    def year_settle(self, state: PropertyState, ctx: YearContext) -> None:
        super().year_settle(state, ctx)
        # Rent and costs are constant within a year (escalation is stage E),
        # so twelve times the monthly figure is the exact yearly total.
        state.cumulative_cashflow += (
            12.0 * self._effective_rent(state)
            - 12.0 * (state.monthly_deductible_costs + state.monthly_own_reserve)
            - state.year_mortgage_paid
            - state.yearly_sondertilgung
        )
        if self.taxation is not None:
            state.year_taxable_result = (
                12.0 * self._effective_rent(state)
                - 12.0 * state.monthly_deductible_costs
                - state.year_interest
                - 12.0 * state.monthly_afa
            )
            ctx.add_taxable_income(state.year_taxable_result)

    def year_grow(self, state: PropertyState, ctx: YearContext) -> None:
        super().year_grow(state, ctx)
        if self.operating_costs is not None:
            rate = self.operating_costs.yearly_increase_rate
            state.monthly_deductible_costs *= 1.0 + rate
            state.monthly_own_reserve *= 1.0 + rate

    def report(self, state: PropertyState, ctx: YearContext) -> dict[str, float]:
        values = super().report(state, ctx)
        values["monthly_rental_income"] = self._effective_rent(state)
        return values | self._extra_report(state)

    def report_initial(self, state: PropertyState) -> dict[str, float]:
        values = super().report_initial(state)
        values["monthly_rental_income"] = self._effective_rent(state)
        return values | self._extra_report(state)

    def _extra_report(self, state: PropertyState) -> dict[str, float]:
        return {
            "monthly_operating_costs": state.monthly_deductible_costs
            + state.monthly_own_reserve,
            "monthly_afa": state.monthly_afa,
            "yearly_taxable_result": state.year_taxable_result,
            "cumulative_cashflow": state.cumulative_cashflow,
        }
