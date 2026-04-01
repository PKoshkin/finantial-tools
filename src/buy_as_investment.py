import pandas as pd


def compute_investment_property_model(
    mortgage_apartment_price: float,
    mortgage_interest_rate: float,
    mortgage_down_payment_rate: float,
    mortgage_total_fees_rate: float,
    mortgage_yearly_repayment_rate: float,
    mortgate_refinancing_years: int,
    rental_income_monthly: float,
    rental_income_yearly_increase_rate: float,
    yearly_apartment_raise_rate: float,
    etf_yearly_return_rate: float,
    initial_capital: float,
    monthly_savings: float,
    years: int,
) -> pd.DataFrame:
    """
    Compute a yearly financial model for buying an investment property (to rent out) vs ETF.

    This models a scenario where:
    - You buy an apartment with a mortgage as an investment
    - You rent it out and receive rental income (assumed tax-free)
    - Surplus cash (rental income - mortgage + monthly savings) is invested into ETF
    - You already own your home, so no personal housing costs are modeled

    Each row represents the end-of-year state. The result includes:
    - total_loan: outstanding mortgage principal at year end
    - property_value: current market value of property
    - property_equity: property_value - total_loan
    - etf_capital: accumulated ETF investments with returns
    - estimated_total_capital: property_equity + etf_capital
    - monthly_rental_income: rental income for that year
    - monthly_mortgage_payment: fixed mortgage payment
    - monthly_surplus: rental_income - mortgage_payment + monthly_savings
    """

    # Type assertions
    assert isinstance(mortgage_apartment_price, (int, float))
    assert isinstance(mortgage_interest_rate, (int, float))
    assert isinstance(mortgage_down_payment_rate, (int, float))
    assert isinstance(mortgage_total_fees_rate, (int, float))
    assert isinstance(mortgage_yearly_repayment_rate, (int, float))
    assert isinstance(mortgate_refinancing_years, int)
    assert isinstance(rental_income_monthly, (int, float))
    assert isinstance(rental_income_yearly_increase_rate, (int, float))
    assert isinstance(yearly_apartment_raise_rate, (int, float))
    assert isinstance(etf_yearly_return_rate, (int, float))
    assert isinstance(initial_capital, (int, float))
    assert isinstance(monthly_savings, (int, float))
    assert isinstance(years, int)

    mortgage_down_payment = mortgage_apartment_price * mortgage_down_payment_rate
    mortgage_total_fees = mortgage_apartment_price * mortgage_total_fees_rate

    # Value validations
    if mortgage_apartment_price < 0:
        raise ValueError("mortgage_apartment_price must be >= 0")
    if mortgage_down_payment < 0:
        raise ValueError("mortgage_down_payment must be >= 0")
    if mortgage_total_fees < 0:
        raise ValueError("mortgage_total_fees must be >= 0")
    if mortgage_down_payment > mortgage_apartment_price:
        raise ValueError("mortgage_down_payment cannot exceed mortgage_apartment_price")
    if initial_capital < mortgage_down_payment + mortgage_total_fees:
        raise ValueError(
            "initial_capital is insufficient to cover down payment and fees"
        )
    if mortgage_interest_rate < 0:
        raise ValueError("mortgage_interest_rate must be >= 0")
    if mortgage_yearly_repayment_rate < 0:
        raise ValueError("mortgage_yearly_repayment_rate must be >= 0")
    if etf_yearly_return_rate <= -1.0:
        raise ValueError("etf_yearly_return_rate must be > -1.0")
    if yearly_apartment_raise_rate <= -1.0:
        raise ValueError("yearly_apartment_raise_rate must be > -1.0")
    if rental_income_yearly_increase_rate <= -1.0:
        raise ValueError("rental_income_yearly_increase_rate must be > -1.0")
    if rental_income_monthly < 0:
        raise ValueError("rental_income_monthly must be >= 0")
    if years <= 0:
        raise ValueError("years must be > 0")
    if mortgate_refinancing_years < 0:
        raise ValueError("mortgate_refinancing_years must be >= 0")

    # Initial loan and capital
    loan_outstanding = mortgage_apartment_price - mortgage_down_payment
    assert loan_outstanding >= 0, "loan_outstanding must be >= 0"

    # Down payment and fees are paid from initial capital upfront
    # Remaining capital goes into ETF
    etf_capital = initial_capital - mortgage_down_payment - mortgage_total_fees

    # Mortgage annuity model: fixed monthly payment until refinancing
    loan_base = loan_outstanding
    fixed_monthly_payment = (
        loan_base * (mortgage_interest_rate + mortgage_yearly_repayment_rate) / 12.0
        if loan_base > 0
        else 0.0
    )

    current_rental_income = rental_income_monthly
    monthly_etf_rate = (1.0 + etf_yearly_return_rate) ** (1.0 / 12.0) - 1.0

    # Year 0 row (initial state before any year passes)
    initial_surplus = current_rental_income - fixed_monthly_payment + monthly_savings
    rows = [
        {
            "year": 0,
            "total_loan": loan_outstanding,
            "property_value": mortgage_apartment_price,
            "property_equity": mortgage_apartment_price - loan_outstanding,
            "etf_capital": etf_capital,
            "estimated_total_capital": (mortgage_apartment_price - loan_outstanding) + etf_capital,
            "monthly_rental_income": current_rental_income,
            "monthly_mortgage_payment": fixed_monthly_payment,
            "monthly_interest_payment": 0,
            "monthly_loan_repayment": 0,
            "monthly_surplus": initial_surplus,
            "monthly_savings": monthly_savings,
        }
    ]

    for year in range(1, years + 1):
        # Property value grows yearly
        property_value = mortgage_apartment_price * (1.0 + yearly_apartment_raise_rate) ** year

        # Track totals for reporting
        total_interest_paid_this_year = 0.0
        total_principal_paid_this_year = 0.0

        # Monthly simulation
        for _ in range(12):
            # Interest calculated from current outstanding principal
            monthly_interest_payment = loan_outstanding * mortgage_interest_rate / 12.0
            monthly_loan_repayment = fixed_monthly_payment - monthly_interest_payment

            total_interest_paid_this_year += monthly_interest_payment
            total_principal_paid_this_year += monthly_loan_repayment

            # Monthly surplus: rental income - mortgage + savings
            monthly_surplus = current_rental_income - fixed_monthly_payment + monthly_savings

            # Invest surplus into ETF (compound existing + add new)
            etf_capital = etf_capital * (1.0 + monthly_etf_rate) + monthly_surplus

            # Reduce loan outstanding
            loan_outstanding -= monthly_loan_repayment
            if loan_outstanding < 0:
                loan_outstanding = 0

        # Average monthly values for reporting
        avg_monthly_interest = total_interest_paid_this_year / 12.0
        avg_monthly_repayment = total_principal_paid_this_year / 12.0

        # Property equity and total capital
        property_equity = property_value - loan_outstanding
        estimated_total_capital = property_equity + etf_capital

        rows.append(
            {
                "year": year,
                "total_loan": loan_outstanding,
                "property_value": property_value,
                "property_equity": property_equity,
                "etf_capital": etf_capital,
                "estimated_total_capital": estimated_total_capital,
                "monthly_rental_income": current_rental_income,
                "monthly_mortgage_payment": fixed_monthly_payment,
                "monthly_interest_payment": avg_monthly_interest,
                "monthly_loan_repayment": avg_monthly_repayment,
                "monthly_surplus": monthly_surplus,
                "monthly_savings": monthly_savings,
            }
        )

        # Prepare next year: rental income grows
        current_rental_income *= 1.0 + rental_income_yearly_increase_rate

        # Recalculate fixed payment on refinancing schedule
        if (
            loan_outstanding > 0
            and mortgate_refinancing_years > 0
            and (year % mortgate_refinancing_years == 0)
        ):
            loan_base = loan_outstanding
            fixed_monthly_payment = (
                loan_base * (mortgage_interest_rate + mortgage_yearly_repayment_rate) / 12.0
            )

    return pd.DataFrame(rows)


def compute_etf_only_model(
    initial_capital: float,
    monthly_savings: float,
    etf_yearly_return_rate: float,
    years: int,
) -> pd.DataFrame:
    """
    Compute a simple ETF-only investment model for comparison.

    All initial capital is invested into ETF, plus monthly savings.
    This serves as the baseline comparison for the investment property model.
    """

    # Validations
    if etf_yearly_return_rate <= -1.0:
        raise ValueError("etf_yearly_return_rate must be > -1.0")
    if years <= 0:
        raise ValueError("years must be > 0")

    monthly_etf_rate = (1.0 + etf_yearly_return_rate) ** (1.0 / 12.0) - 1.0
    etf_capital = initial_capital

    rows = [
        {
            "year": 0,
            "etf_capital": etf_capital,
            "estimated_total_capital": etf_capital,
            "monthly_savings": monthly_savings,
        }
    ]

    for year in range(1, years + 1):
        # Monthly compounding with savings contributions
        for _ in range(12):
            etf_capital = etf_capital * (1.0 + monthly_etf_rate) + monthly_savings

        rows.append(
            {
                "year": year,
                "etf_capital": etf_capital,
                "estimated_total_capital": etf_capital,
                "monthly_savings": monthly_savings,
            }
        )

    return pd.DataFrame(rows)
