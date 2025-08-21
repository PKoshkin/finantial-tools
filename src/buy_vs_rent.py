import pandas as pd


def compute_finantial_model(
    yearly_inflation_rate: float,
    yearly_apartment_raise_rate: float,
    mortgage_apartment_price: float,
    mortgage_interest_rate: float,
    mortgage_down_payment_rate: float,
    mortgage_total_fees_rate: float,
    mortgage_yearly_repayment_rate: float,
    mortgate_refinancing_years: int,
    etf_yearly_return_rate: float,
    cold_rent_monthly_cost: float,
    cold_rent_yearly_increase_rate: float,
    initial_capital: float,
    monthly_net_income: float,
    monthly_spending: float,
    yearly_income_increase_rate: float,
    years: int,
) -> pd.DataFrame:
    """
    Compute a yearly financial model when simultaneously owning (with a mortgage) and renting.

    Each row represents the end-of-year state. The result includes at least:
    - total_loan: outstanding mortgage principal at year end
    - estimated_total_capital: invested capital + property equity at year end
    - monthly_apartment_spend: average monthly spending on rent + mortgage for that year

    Additional assumptions for this version:
    - monthly_spending grows monthly by inflation
    - leftover cash after apartment spending and monthly_spending is invested into ETF monthly
    """

    # Type assertions
    assert isinstance(yearly_inflation_rate, (int, float))
    assert isinstance(yearly_apartment_raise_rate, (int, float))
    assert isinstance(mortgage_apartment_price, (int, float))
    assert isinstance(mortgage_interest_rate, (int, float))
    assert isinstance(mortgage_down_payment_rate, (int, float))
    assert isinstance(mortgage_total_fees_rate, (int, float))
    assert isinstance(mortgage_yearly_repayment_rate, (int, float))
    assert isinstance(mortgate_refinancing_years, int)
    assert isinstance(etf_yearly_return_rate, (int, float))
    assert isinstance(cold_rent_monthly_cost, (int, float))
    assert isinstance(cold_rent_yearly_increase_rate, (int, float))
    assert isinstance(initial_capital, (int, float))
    assert isinstance(monthly_net_income, (int, float))
    assert isinstance(monthly_spending, (int, float))
    assert isinstance(yearly_income_increase_rate, (int, float))
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
    if yearly_inflation_rate <= -1.0:
        raise ValueError("yearly_inflation_rate must be > -1.0")
    if yearly_apartment_raise_rate <= -1.0:
        raise ValueError("yearly_apartment_raise_rate must be > -1.0")
    if cold_rent_yearly_increase_rate <= -1.0:
        raise ValueError("cold_rent_yearly_increase_rate must be > -1.0")
    if yearly_income_increase_rate <= -1.0:
        raise ValueError("yearly_income_increase_rate must be > -1.0")
    if cold_rent_monthly_cost < 0:
        raise ValueError("cold_rent_monthly_cost must be >= 0")
    if monthly_net_income < 0:
        raise ValueError("monthly_net_income must be >= 0")
    if monthly_spending < 0:
        raise ValueError("monthly_spending must be >= 0")
    if years <= 0:
        raise ValueError("years must be > 0")
    if mortgate_refinancing_years < 0:
        raise ValueError("mortgate_refinancing_years must be >= 0")

    # Initial loan and capital
    loan_outstanding = mortgage_apartment_price - mortgage_down_payment
    assert loan_outstanding >= 0, "loan_outstanding must be >= 0"

    # Down payment and fees are assumed to be paid from initial capital upfront
    invested_capital = initial_capital - mortgage_down_payment - mortgage_total_fees
    etf_capital = initial_capital - mortgage_down_payment - mortgage_total_fees

    # Guard: if initial capital cannot cover down payment and fees, allow negative cash
    # (represents borrowing from other sources). Keep the model simple and explicit.

    # Mortgage annuity model: yearly payment is (interest + amortization) * base.
    # Base is reset on refinancing years.
    loan_base = loan_outstanding

    current_monthly_income = monthly_net_income
    current_monthly_rent = cold_rent_monthly_cost
    current_monthly_spending = monthly_spending

    monthly_inflation_rate = (1.0 + yearly_inflation_rate) ** (1.0 / 12.0) - 1.0
    monthly_etf_rate = (1.0 + etf_yearly_return_rate) ** (1.0 / 12.0) - 1.0

    rows = [
            {
                "year": 0,
                "total_loan": 0,
                "estimated_total_capital": initial_capital,
                "monthly_interest_payment": 0,
                "monthly_loan_repayment": 0,
                "monthly_rent": current_monthly_rent,
                "monthly_apartment_spend": current_monthly_rent,
                "monthly_spending": current_monthly_spending,
                "monthly_income": current_monthly_income,
                "monthly_leftover": current_monthly_income - current_monthly_spending - current_monthly_rent,
                "invested_capital": initial_capital,
                "property_value": 0,
                "property_equity": 0,
            }
    ]
    for year in range(1, years + 1):
        # Property value grows with inflation as a proxy
        property_value = mortgage_apartment_price * (1.0 + yearly_apartment_raise_rate) ** year

        monthly_interest_payment = loan_base * mortgage_interest_rate / 12.0
        monthly_loan_repayment = loan_base * mortgage_yearly_repayment_rate / 12.0
        monthly_apartment_spend = current_monthly_rent + monthly_interest_payment + monthly_loan_repayment

        # Cashflow and investments (ETF) done monthly
        for _ in range(12):
            monthly_leftover = (
                current_monthly_income
                - monthly_apartment_spend
                - current_monthly_spending
            )
            invested_capital += monthly_leftover
            etf_capital = etf_capital * (1.0 + monthly_etf_rate) + monthly_leftover
            # Update monthly spending with inflation for next month
            current_monthly_spending *= 1.0 + monthly_inflation_rate
            loan_outstanding -= monthly_loan_repayment

        # Estimated total capital = invested capital + property equity
        property_equity = property_value - loan_outstanding
        estimated_total_capital =  property_equity + etf_capital

        rows.append(
            {
                "year": year,
                "total_loan": loan_outstanding,
                "estimated_total_capital": estimated_total_capital,
                "monthly_interest_payment": monthly_interest_payment,
                "monthly_loan_repayment": monthly_loan_repayment,
                "monthly_rent": current_monthly_rent,
                "monthly_apartment_spend": monthly_apartment_spend,
                "monthly_spending": current_monthly_spending,
                "monthly_income": current_monthly_income,
                "monthly_leftover": monthly_leftover,
                "invested_capital": invested_capital,
                "property_value": property_value,
                "property_equity": property_equity,
            }
        )

        # Prepare next year values (income/rent growth and possible refinancing)
        current_monthly_income *= 1.0 + yearly_income_increase_rate
        current_monthly_rent *= 1.0 + cold_rent_yearly_increase_rate
        # monthly spending already compounded intra-year; keep its last value going forward

        # Recalculate loan base on refinancing schedule
        if (
            loan_outstanding > 0
            and (year % mortgate_refinancing_years == 0)
        ):
            loan_base = loan_outstanding

        # Monthly spend on apartments for next year
        monthly_interest_payment = loan_base * mortgage_interest_rate / 12.0
        monthly_loan_repayment = loan_base * mortgage_yearly_repayment_rate / 12.0
        monthly_apartment_spend = current_monthly_rent + monthly_interest_payment + monthly_loan_repayment

    return pd.DataFrame(rows)
