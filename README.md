# finantial-tools

Personal financial planning tools built around **one unified financial model**.

## Structure

```
src/financial_model.py      # the single core model (Scenario + simulate)
tests/                      # test suite (pytest)
tests/golden/               # pinned outputs of the pre-refactoring legacy models
notebook/                   # analyses; each notebook = the same core, different parameters
```

## The unified model

`src/financial_model.py` contains a monthly-resolution simulation engine. A
scenario is described by a single `Scenario` dataclass; `simulate(scenario)`
returns a yearly `DataFrame` with the forecast of all financial indicators
(loan, property value/equity, ETF capital gross/after tax, total wealth,
monthly cash flows, interest paid, ...).

Every feature is off by default, so a scenario only sets the factors it cares
about — and all factors can be combined in one scenario:

| Factor | Parameters |
|---|---|
| Household cash flow | `monthly_net_income`, `yearly_income_increase_rate`, `monthly_spending`, `yearly_inflation_rate` |
| Renting your home | `monthly_rent`, `yearly_rent_increase_rate` |
| Buying property | `apartment_price`, `yearly_apartment_price_increase_rate`, `mortgage_down_payment_rate`, `mortgage_fees_rate` |
| Mortgage (annuity) | `mortgage_interest_rate`, `mortgage_yearly_repayment_rate`, `mortgage_refinancing_years`, `sondertilgung_yearly_rate` |
| Renting the property out | `monthly_rental_income`, `yearly_rental_income_increase_rate` |
| ETF investing | `etf_yearly_return_rate`, `etf_tax_rate`, `etf_yearly_withdrawal_rate` |

```python
from src.financial_model import Scenario, simulate, replace

scenario = Scenario(
    years=30,
    initial_capital=100_000,
    monthly_net_income=5_000,
    monthly_spending=2_000,
    yearly_inflation_rate=0.02,
    apartment_price=450_000,
    mortgage_interest_rate=0.038,
    mortgage_down_payment_rate=0.10,
    mortgage_fees_rate=0.10,
    mortgage_yearly_repayment_rate=0.015,
    mortgage_refinancing_years=10,
    sondertilgung_yearly_rate=0.05,
    etf_yearly_return_rate=0.07,
    etf_tax_rate=0.26375,
)
df = simulate(scenario)

# scenario sweeps
df_no_st = simulate(replace(scenario, sondertilgung_yearly_rate=0.0))
```

Model semantics (year-0 row, tax/cost-basis treatment, Sondertilgung reserve,
withdrawal ordering) are documented in the module docstring of
[src/financial_model.py](src/financial_model.py).

## Notebooks

All notebooks import the same core and differ only in parameters:

- `buy_vs_rent.ipynb` — buy your home with a mortgage vs rent + invest in ETF
- `sondertilgung_vs_etf.ipynb` — extra mortgage repayments vs ETF, breakeven ETF rate, early-repayment penalty, MSCI World stats
- `buy_as_investment_vs_etf.ipynb` — buy-to-let vs ETF-only
- `etf_withdrawal.ipynb` — ETF accumulation with a yearly percentage withdrawal

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

The suite has two layers:

1. **Legacy equivalence** — before the refactoring the repo contained three
   separate models; their outputs for the notebook scenarios are pinned in
   `tests/golden/` and the unified model reproduces them wherever semantics
   intentionally coincide (the deliberate deviations — including two legacy
   money-leak bugs that were fixed — are documented in
   [tests/test_financial_model.py](tests/test_financial_model.py)).
2. **Invariants** — cash conservation, closed-form annuity amortization and
   ETF compounding, tax-on-gains-only, Sondertilgung caps, refinancing
   behavior, validation.
