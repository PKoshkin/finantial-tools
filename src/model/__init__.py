"""A component-based personal financial model.

A forecast is a `FinancialModel` plus the components it is made of. Every
component type influences the simulation in its own way; some can appear at
most once (the home you live in, the rent you pay), others any number of times
(buy-to-let flats, salaries, portfolios).

    from src.model import FinancialModel, Salary, Spending, Home, Mortgage, Etf

    model = FinancialModel(
        components=[
            Salary(monthly_amount=5_000, yearly_increase_rate=0.02),
            Spending(monthly_amount=2_000),               # follows the model's inflation
            Home(
                price=450_000,
                yearly_price_increase_rate=0.03,
                mortgage=Mortgage(
                    interest_rate=0.038,
                    down_payment_rate=0.10,
                    fees_rate=0.10,
                    yearly_repayment_rate=0.015,
                    refinancing_years=10,
                    sondertilgung_yearly_rate=0.05,
                ),
            ),
            Etf(initial_capital=100_000, yearly_return_rate=0.07, tax_rate=0.26375),
        ],
        years=30,
        yearly_inflation_rate=0.02,
    )
    df = model.run()

    # sweeps return new models, never mutate this one
    without_extra_repayments = model.replaced("home.mortgage", sondertilgung_yearly_rate=0.0)

Model semantics — the details that decide what the numbers mean:

- Monthly simulation, yearly reporting. Portfolios compound monthly:
  `capital = capital * (1 + r_m) + contribution`, i.e. growth comes first.
- Every yearly rate names how it becomes a monthly one, via `Compounding`.
  A mortgage's Sollzins is `NOMINAL` (`rate / 12` a month, so 3.8% a year
  accrues to 3.867% effective — the loan the bank actually sells); a quoted
  market return or inflation figure is `EFFECTIVE` (`(1 + rate) ** (1/12) - 1`,
  so twelve months reproduce it exactly). Both are overridable per component;
  the defaults are the conventions every pinned figure was computed with.
- Yearly escalations — salary raises, rent increases, property appreciation —
  are applied once a year and never converted to a monthly rate, so no
  compounding convention applies to them.
- Row 0 is the state immediately after the t=0 purchases: the loan is
  outstanding, the down payment has become property equity, the purchase fees
  are already sunk.
- A mortgage is an annuity: the payment is `loan * (interest + repayment) / 12`,
  recomputed from the outstanding principal every `refinancing_years`, and it
  stops as soon as the loan is repaid — the freed cash flows into the sink.
- Sondertilgung: a fraction of the ORIGINAL loan is withheld from the leftover
  every month and applied to the principal at year end. Whatever the loan can
  no longer use goes into the sink instead.
- Portfolio taxation: capital gains tax on `capital - cost basis`, reported in
  the `*_after_tax` columns; the gross capital keeps compounding untaxed until
  sale. Selling reduces the cost basis proportionally to the sold share of the
  portfolio (average-cost method).
- A yearly withdrawal is taken at year end, after the Sondertilgung, taxed on
  its gain part, and considered consumed rather than reinvested.
- Spending grows monthly with inflation; salaries, rent, rental income and a
  flat's operating costs grow once a year, after the row for that year has
  been emitted.
- Rental taxation: a flat with `RentalTaxation` publishes its yearly
  Einkünfte aus Vermietung und Verpachtung (rent − deductible operating costs
  − loan interest − AfA on `(price + fees) * building_share`) to the
  `GermanHousehold`, which re-assesses the year's income tax with it — the
  cash effect rides the July refund of the following year. Losses offset wage
  income within their year only (no Verlustvortrag).
- Monthly flow columns (interest, repayment, leftover, spending, ...) are
  averages over the 12 months of that year.
- `monthly_leftover` is the household surplus before anything is withheld from
  it; `monthly_invested` is the part that reached the portfolios. The gap is
  exactly `yearly_sondertilgung / 12`, the money the principal took — so a plan
  can show a healthy leftover while the portfolio shrinks. A reserve the loan
  could not absorb comes back to the sink at year end and counts as invested.
  At year 0 no month has run yet and the two are equal.
- The sink may go NEGATIVE when the cash flow cannot fund the plan: the
  uncovered deficit keeps compounding at the portfolio's rate, i.e. an implicit
  loan. Check feasibility (e.g. `(df.etf_capital < 0).any()`) before trusting a
  comparison.
"""

from .component import (
    Cardinality,
    CashDraw,
    CashFlow,
    CashPhase,
    Compounding,
    FinancialComponent,
    ModelParams,
    MonthContext,
    SetupContext,
    YearContext,
)
from .components import (
    REMAINDER,
    Earner,
    Etf,
    GermanHousehold,
    Home,
    InvestmentProperty,
    Mortgage,
    OperatingCosts,
    PrivateHealthInsurance,
    Rent,
    RentalTaxation,
    Salary,
    Spending,
)
from .german_tax import TaxSystem
from .model import FinancialModel
from .reporting import (
    FLAT_COLUMNS,
    calculate_early_repayment_penalty,
    flat,
    payoff_year,
)

__all__ = [
    "FLAT_COLUMNS",
    "REMAINDER",
    "Cardinality",
    "CashDraw",
    "CashFlow",
    "CashPhase",
    "Compounding",
    "Earner",
    "Etf",
    "FinancialComponent",
    "FinancialModel",
    "GermanHousehold",
    "Home",
    "InvestmentProperty",
    "ModelParams",
    "MonthContext",
    "Mortgage",
    "OperatingCosts",
    "PrivateHealthInsurance",
    "Rent",
    "RentalTaxation",
    "Salary",
    "SetupContext",
    "Spending",
    "TaxSystem",
    "YearContext",
    "calculate_early_repayment_penalty",
    "flat",
    "payoff_year",
]
