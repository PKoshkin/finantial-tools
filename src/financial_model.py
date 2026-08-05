"""Unified financial model.

One monthly-resolution simulation engine that covers every scenario the
repository analyses. A scenario is described by a single `Scenario` dataclass;
`simulate()` returns a yearly DataFrame with the forecast of all financial
indicators. Different notebooks use the same core with different parameters:

- Buy (own home) vs rent:      apartment_price > 0 vs apartment_price == 0
- Buy-to-let vs ETF:           monthly_rental_income > 0 vs apartment_price == 0
- Sondertilgung vs ETF:        sweep sondertilgung_yearly_rate
- ETF with yearly withdrawal:  etf_yearly_withdrawal_rate > 0

Model semantics (documented deviations from the legacy per-notebook models):

- Monthly simulation. ETF compounds monthly: capital = capital * (1 + r_m) + contribution.
  Monthly rates are geometric: (1 + yearly) ** (1/12) - 1.
- Year 0 row is the state immediately AFTER the purchase transaction: the loan
  is outstanding, the down payment has become property equity and the purchase
  fees are already sunk. (The legacy buy_vs_rent model reported a pre-purchase
  year 0 with total_loan = 0 and capital = initial_capital.)
- The mortgage is an annuity: fixed monthly payment = loan * (interest + repayment) / 12,
  recomputed from the outstanding principal every `mortgage_refinancing_years`.
  The payment stops as soon as the loan is repaid — the freed cash flows into
  the ETF. (The legacy buy_as_investment model kept deducting the payment
  forever; that money simply vanished.)
- Sondertilgung: a fraction of the ORIGINAL loan is reserved monthly from the
  leftover and applied to the principal at year end. Whatever part of the
  reserve cannot be applied (loan smaller than the reserve) is invested into
  the ETF at year end. (In the legacy model the unused reserve vanished.)
- ETF taxation: capital gains tax on (capital - cost basis), applied in the
  etf_capital_after_tax / estimated_total_capital columns; the gross capital
  keeps compounding untaxed until sale. Negative monthly contributions (selling
  to cover a cash deficit) reduce the cost basis proportionally to the sold
  share of the portfolio. (The legacy model reduced the basis by the full sale
  amount, overstating the future tax.)
- Yearly ETF withdrawal (if etf_yearly_withdrawal_rate > 0) is taken at year
  end, after the Sondertilgung. Capital gains tax is applied to the gain part
  of the withdrawal; the net amount is reported (and considered consumed, not
  reinvested).
- Spending grows monthly with inflation; income, own rent and rental income
  grow yearly (at year end).
- Monthly flow columns (interest, repayment, leftover, spending, ...) are
  averages over the 12 months of that year.
- The ETF capital may go NEGATIVE when the household cash flow cannot fund the
  spending/rent/mortgage/Sondertilgung: the uncovered deficit keeps compounding
  at the ETF rate, i.e. an implicit loan at that rate. Check scenario
  feasibility (e.g. `(df.etf_capital < 0).any()`) before trusting comparisons.
"""

from dataclasses import dataclass, replace  # noqa: F401  (replace re-exported for notebooks)

import pandas as pd


@dataclass(frozen=True)
class Scenario:
    """Full parameter set of the unified financial model.

    Every feature is off by default (rate/amount 0), so a scenario only sets
    the factors it cares about. `apartment_price == 0` means no purchase at
    all; `monthly_rental_income > 0` means the property is rented out
    (buy-to-let); `monthly_rent > 0` means you pay rent for your own housing
    (possible simultaneously with a purchase).
    """

    years: int
    initial_capital: float

    # --- Household cash flow ---
    monthly_net_income: float = 0.0
    yearly_income_increase_rate: float = 0.0
    monthly_spending: float = 0.0
    yearly_inflation_rate: float = 0.0  # drives spending growth
    monthly_rent: float = 0.0  # cold rent for your own housing
    yearly_rent_increase_rate: float = 0.0

    # --- Property purchase (apartment_price == 0 -> no purchase) ---
    apartment_price: float = 0.0
    yearly_apartment_price_increase_rate: float = 0.0
    monthly_rental_income: float = 0.0  # rent you receive (buy-to-let)
    yearly_rental_income_increase_rate: float = 0.0

    # --- Mortgage ---
    mortgage_interest_rate: float = 0.0
    mortgage_down_payment_rate: float = 0.0
    mortgage_fees_rate: float = 0.0  # notary, taxes, agent: % of price, sunk
    mortgage_yearly_repayment_rate: float = 0.0  # initial Tilgung rate
    mortgage_refinancing_years: int = 0  # 0 -> annuity never recomputed
    sondertilgung_yearly_rate: float = 0.0  # % of ORIGINAL loan per year

    # --- ETF ---
    etf_yearly_return_rate: float = 0.0
    etf_tax_rate: float = 0.0  # capital gains tax (e.g. Abgeltungssteuer)
    etf_yearly_withdrawal_rate: float = 0.0  # yearly extraction at year end

    # --- Reporting only ---
    safe_withdrawal_rate: float = 0.03  # for monthly_costs_not_covered_by_safe_withdrawal

    def __post_init__(self) -> None:
        if not isinstance(self.years, int):
            raise TypeError("years must be an int")
        if not isinstance(self.mortgage_refinancing_years, int):
            raise TypeError("mortgage_refinancing_years must be an int")
        for name in (
            "initial_capital",
            "monthly_net_income",
            "monthly_spending",
            "monthly_rent",
            "apartment_price",
            "monthly_rental_income",
            "mortgage_interest_rate",
            "mortgage_down_payment_rate",
            "mortgage_fees_rate",
            "mortgage_yearly_repayment_rate",
            "sondertilgung_yearly_rate",
            "safe_withdrawal_rate",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
        for name in (
            "yearly_income_increase_rate",
            "yearly_inflation_rate",
            "yearly_rent_increase_rate",
            "yearly_apartment_price_increase_rate",
            "yearly_rental_income_increase_rate",
            "etf_yearly_return_rate",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if value <= -1.0:
                raise ValueError(f"{name} must be > -1.0")
        if self.years <= 0:
            raise ValueError("years must be > 0")
        if self.mortgage_refinancing_years < 0:
            raise ValueError("mortgage_refinancing_years must be >= 0")
        if not 0.0 <= self.mortgage_down_payment_rate <= 1.0:
            raise ValueError("mortgage_down_payment_rate must be within [0, 1]")
        if not 0.0 <= self.etf_tax_rate < 1.0:
            raise ValueError("etf_tax_rate must be within [0, 1)")
        if not 0.0 <= self.etf_yearly_withdrawal_rate <= 1.0:
            raise ValueError("etf_yearly_withdrawal_rate must be within [0, 1]")
        if self.initial_capital < self.down_payment + self.purchase_fees:
            raise ValueError(
                "initial_capital is insufficient to cover down payment and fees"
            )

    # Derived amounts (properties so notebooks can print them)
    @property
    def down_payment(self) -> float:
        return self.apartment_price * self.mortgage_down_payment_rate

    @property
    def purchase_fees(self) -> float:
        return self.apartment_price * self.mortgage_fees_rate

    @property
    def initial_loan(self) -> float:
        return self.apartment_price - self.down_payment

    @property
    def initial_monthly_mortgage_payment(self) -> float:
        return _annuity_payment(self.initial_loan, self)

    @property
    def max_yearly_sondertilgung(self) -> float:
        return self.initial_loan * self.sondertilgung_yearly_rate


def _annuity_payment(loan_base: float, scenario: Scenario) -> float:
    if loan_base <= 0:
        return 0.0
    return (
        loan_base
        * (scenario.mortgage_interest_rate + scenario.mortgage_yearly_repayment_rate)
        / 12.0
    )


def _monthly_rate(yearly_rate: float) -> float:
    return (1.0 + yearly_rate) ** (1.0 / 12.0) - 1.0


def _sell_etf(capital: float, cost_basis: float, amount: float) -> float:
    """Return the new cost basis after selling `amount` from the portfolio.

    The basis is reduced proportionally to the sold share of the portfolio
    (average-cost method). If the portfolio is empty/negative the basis is
    left unchanged (degenerate case: the model is in uncovered debt).
    """
    if capital <= 0 or amount <= 0:
        return cost_basis
    sold_fraction = min(amount / capital, 1.0)
    return cost_basis * (1.0 - sold_fraction)


def simulate(scenario: Scenario) -> pd.DataFrame:
    """Run the unified model and return a yearly end-of-year forecast.

    Row 0 is the state immediately after the purchase transaction (or simply
    the initial state when nothing is bought). Rows 1..years are end-of-year
    states. See the module docstring for exact semantics.
    """
    s = scenario

    loan_outstanding = s.initial_loan
    fixed_monthly_payment = _annuity_payment(loan_outstanding, s)

    etf_capital = s.initial_capital - s.down_payment - s.purchase_fees
    etf_cost_basis = max(etf_capital, 0.0)
    invested_capital = s.initial_capital  # cash that passed through, no returns

    current_monthly_income = s.monthly_net_income
    current_monthly_spending = s.monthly_spending
    current_monthly_rent = s.monthly_rent
    current_monthly_rental_income = s.monthly_rental_income

    monthly_inflation = _monthly_rate(s.yearly_inflation_rate)
    monthly_etf_rate = _monthly_rate(s.etf_yearly_return_rate)

    sondertilgung_monthly_reserve = s.max_yearly_sondertilgung / 12.0
    cumulative_interest_paid = 0.0

    def etf_after_tax(capital: float, basis: float) -> float:
        gain = max(0.0, capital - basis)
        return capital - gain * s.etf_tax_rate

    def make_row(**overrides) -> dict:
        row = {
            "year": 0,
            "total_loan": loan_outstanding,
            "property_value": 0.0,
            "property_equity": 0.0,
            "etf_capital": etf_capital,
            "etf_cost_basis": etf_cost_basis,
            "etf_capital_after_tax": etf_after_tax(etf_capital, etf_cost_basis),
            "estimated_total_capital": 0.0,
            "invested_capital": invested_capital,
            "monthly_income": current_monthly_income,
            "monthly_rental_income": current_monthly_rental_income,
            "monthly_spending": current_monthly_spending,
            "monthly_rent": current_monthly_rent,
            "monthly_interest_payment": 0.0,
            "monthly_loan_repayment": 0.0,
            "monthly_mortgage_payment": 0.0,
            "monthly_housing_cost": 0.0,
            "monthly_leftover": 0.0,
            "yearly_sondertilgung": 0.0,
            "cumulative_interest_paid": cumulative_interest_paid,
            "yearly_etf_withdrawal_gross": 0.0,
            "yearly_etf_withdrawal_net": 0.0,
            "monthly_costs_not_covered_by_safe_withdrawal": 0.0,
        }
        row.update(overrides)
        return row

    # --- Year 0: state right after the purchase ---
    initial_property_equity = s.apartment_price - loan_outstanding
    initial_after_tax = etf_after_tax(etf_capital, etf_cost_basis)
    initial_housing_cost = current_monthly_rent + fixed_monthly_payment
    rows = [
        make_row(
            property_value=s.apartment_price,
            property_equity=initial_property_equity,
            estimated_total_capital=initial_property_equity + initial_after_tax,
            monthly_mortgage_payment=fixed_monthly_payment,
            monthly_housing_cost=initial_housing_cost,
            monthly_leftover=(
                current_monthly_income
                + current_monthly_rental_income
                - current_monthly_spending
                - initial_housing_cost
            ),
            monthly_costs_not_covered_by_safe_withdrawal=(
                (current_monthly_spending + initial_housing_cost)
                - initial_after_tax * s.safe_withdrawal_rate / 12.0
            ),
        )
    ]

    for year in range(1, s.years + 1):
        property_value = (
            s.apartment_price * (1.0 + s.yearly_apartment_price_increase_rate) ** year
        )

        year_interest = 0.0
        year_principal = 0.0
        year_spending = 0.0
        year_leftover = 0.0
        year_mortgage_paid = 0.0
        sondertilgung_reserved = 0.0

        for _ in range(12):
            # Mortgage annuity payment (stops once the loan is repaid)
            if loan_outstanding > 0:
                monthly_interest = loan_outstanding * s.mortgage_interest_rate / 12.0
                monthly_principal = min(
                    fixed_monthly_payment - monthly_interest, loan_outstanding
                )
                mortgage_payment = monthly_interest + monthly_principal
                loan_outstanding -= monthly_principal
                if loan_outstanding <= 0:
                    loan_outstanding = 0.0
                    fixed_monthly_payment = 0.0
                reserve = sondertilgung_monthly_reserve
            else:
                monthly_interest = 0.0
                monthly_principal = 0.0
                mortgage_payment = 0.0
                reserve = 0.0

            monthly_leftover = (
                current_monthly_income
                + current_monthly_rental_income
                - current_monthly_spending
                - current_monthly_rent
                - mortgage_payment
            )
            etf_contribution = monthly_leftover - reserve
            etf_capital = etf_capital * (1.0 + monthly_etf_rate) + etf_contribution
            if etf_contribution >= 0:
                etf_cost_basis += etf_contribution
            else:
                etf_cost_basis = _sell_etf(
                    etf_capital + (-etf_contribution), etf_cost_basis, -etf_contribution
                )

            year_interest += monthly_interest
            year_principal += monthly_principal
            year_mortgage_paid += mortgage_payment
            year_spending += current_monthly_spending
            year_leftover += monthly_leftover
            invested_capital += monthly_leftover
            sondertilgung_reserved += reserve

            current_monthly_spending *= 1.0 + monthly_inflation

        # --- Year end: apply Sondertilgung from the reserved cash ---
        actual_sondertilgung = min(sondertilgung_reserved, loan_outstanding)
        if actual_sondertilgung > 0:
            loan_outstanding -= actual_sondertilgung
            if loan_outstanding <= 0:
                loan_outstanding = 0.0
                fixed_monthly_payment = 0.0
        unused_reserve = sondertilgung_reserved - actual_sondertilgung
        if unused_reserve > 0:
            # The reserve that the loan no longer needs goes into the ETF
            etf_capital += unused_reserve
            etf_cost_basis += unused_reserve

        # --- Year end: ETF withdrawal (consumed, not reinvested) ---
        withdrawal_gross = 0.0
        withdrawal_net = 0.0
        if s.etf_yearly_withdrawal_rate > 0 and etf_capital > 0:
            withdrawal_gross = etf_capital * s.etf_yearly_withdrawal_rate
            gain_fraction = max(0.0, 1.0 - etf_cost_basis / etf_capital)
            withdrawal_net = withdrawal_gross * (
                1.0 - gain_fraction * s.etf_tax_rate
            )
            etf_cost_basis = _sell_etf(etf_capital, etf_cost_basis, withdrawal_gross)
            etf_capital -= withdrawal_gross

        cumulative_interest_paid += year_interest

        avg_spending = year_spending / 12.0
        avg_mortgage_payment = year_mortgage_paid / 12.0
        avg_housing_cost = current_monthly_rent + avg_mortgage_payment
        after_tax = etf_after_tax(etf_capital, etf_cost_basis)
        property_equity = property_value - loan_outstanding

        rows.append(
            make_row(
                year=year,
                property_value=property_value,
                property_equity=property_equity,
                estimated_total_capital=property_equity + after_tax,
                monthly_spending=avg_spending,
                monthly_interest_payment=year_interest / 12.0,
                monthly_loan_repayment=year_principal / 12.0,
                monthly_mortgage_payment=avg_mortgage_payment,
                monthly_housing_cost=avg_housing_cost,
                monthly_leftover=year_leftover / 12.0,
                yearly_sondertilgung=actual_sondertilgung,
                yearly_etf_withdrawal_gross=withdrawal_gross,
                yearly_etf_withdrawal_net=withdrawal_net,
                monthly_costs_not_covered_by_safe_withdrawal=(
                    (avg_spending + avg_housing_cost)
                    - after_tax * s.safe_withdrawal_rate / 12.0
                ),
            )
        )

        # --- Prepare next year: growth and refinancing ---
        current_monthly_income *= 1.0 + s.yearly_income_increase_rate
        current_monthly_rent *= 1.0 + s.yearly_rent_increase_rate
        current_monthly_rental_income *= 1.0 + s.yearly_rental_income_increase_rate

        if (
            loan_outstanding > 0
            and s.mortgage_refinancing_years > 0
            and year % s.mortgage_refinancing_years == 0
        ):
            fixed_monthly_payment = _annuity_payment(loan_outstanding, s)

    return pd.DataFrame(rows)


def payoff_year(model: pd.DataFrame) -> int | None:
    """First year (>= 1) in which the mortgage is fully repaid, or None."""
    paid_off = model[(model["year"] > 0) & (model["total_loan"] <= 0)]
    return int(paid_off["year"].iloc[0]) if not paid_off.empty else None


def calculate_early_repayment_penalty(
    remaining_principal: float,
    loan_rate: float,
    market_rate: float,
    remaining_years: float,
) -> float:
    """Approximate Vorfälligkeitsentschädigung (early repayment penalty).

    After 10 years from loan start the penalty is 0 (with 6 months notice).
    Before that, the penalty compensates the bank for the interest rate
    differential over the remaining Zinsbindung period.
    """
    if remaining_years <= 0:
        return 0.0
    return remaining_principal * max(0.0, loan_rate - market_rate) * remaining_years
