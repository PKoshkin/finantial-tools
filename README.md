# finantial-tools

Personal financial planning tools built around **one model made of components**.

## Structure

```
src/model/                  # the model: FinancialModel + the component types
tests/                      # test suite (pytest)
tests/golden/               # pinned forecasts the model must reproduce
scripts/                    # notebook execution and output diffing
notebook/                   # analyses; each notebook = the same model, different components
artifacts/baseline/         # reference notebook run
```

## The model

A forecast is a `FinancialModel` and the components it is built from. The model
holds only what every component shares — the horizon, the starting age,
inflation. Everything else (income, spending, rent, property, portfolios) is a
component, and each component type influences the simulation in its own way.

```python
from src.model import Etf, FinancialModel, Home, Mortgage, Salary, Spending

model = FinancialModel(
    components=[
        Salary(monthly_amount=5_000, yearly_increase_rate=0.02),
        Spending(monthly_amount=2_000),              # follows the model's inflation
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
    start_age=35,
    yearly_inflation_rate=0.02,
)
df = model.run()
```

### Model parameters

| Parameter | Meaning |
|---|---|
| `years` | horizon; the forecast has `years + 1` rows |
| `start_age` | optional; adds an `age` column, changes nothing else |
| `yearly_inflation_rate` | default growth for `Spending` |
| `safe_withdrawal_rate` | reporting only, feeds `monthly_costs_not_covered_by_safe_withdrawal` |

### Components

| Component | How many | Parameters |
|---|---|---|
| `Salary` | any | `monthly_amount`, `yearly_increase_rate` |
| `Spending` | any | `monthly_amount`, `yearly_increase_rate` (`None` → the model's inflation), `compounding` |
| `Rent` — your own housing | 0 or 1 | `monthly_amount`, `yearly_increase_rate` |
| `Home` — the place you live in | 0 or 1 | `price`, `yearly_price_increase_rate`, `mortgage` |
| `InvestmentProperty` — bought to rent out | any | the above plus `monthly_rental_income`, `yearly_rental_income_increase_rate` |
| `Etf` — a portfolio | any | `initial_capital`, `yearly_return_rate`, `tax_rate`, `monthly_investment`, `yearly_withdrawal_rate`, `compounding` |

Every component also takes a `name`, which prefixes its own result columns.

`Mortgage` is not a component but a value object nested in a property
(`interest_rate`, `down_payment_rate`, `fees_rate`, `yearly_repayment_rate`,
`refinancing_years`, `sondertilgung_yearly_rate`, `compounding`), because a loan
here always belongs to one property and is derived from its price.

Rates of return are per component: two portfolios can grow at different rates
and be taxed differently.

### How a yearly rate becomes a monthly one

Every yearly rate that the simulation needs monthly says which convention it is
quoted in, via `Compounding` — the two coexist because that is how the real
figures are quoted:

| | monthly | 3.8% a year works out to | default for |
|---|---|---|---|
| `NOMINAL` | `rate / 12` | 3.867% effective | `Mortgage` — a Sollzins accrues monthly, which is the loan the bank sells |
| `EFFECTIVE` | `(1 + rate) ** (1/12) - 1` | 3.800% effective | `Etf`, `Spending` — a quoted return or inflation figure already is the annual figure |

```python
Mortgage(interest_rate=0.038, compounding=Compounding.NOMINAL)    # default
Etf(yearly_return_rate=0.07, compounding=Compounding.EFFECTIVE)   # default
```

A mortgage's convention also splits the annuity, so that `yearly_repayment_rate`
always means the share of the loan repaid in the first year: the payment is the
month's interest plus `loan * yearly_repayment_rate / 12`, which for the nominal
default is the familiar `loan * (interest_rate + yearly_repayment_rate) / 12`.

The defaults are the conventions every pinned figure was computed with, so
overriding one deliberately moves the numbers. On the 30-year model above:
switching the mortgage to `EFFECTIVE` cuts total interest from €110.1k to
€108.5k and leaves more principal outstanding (the annuity shrinks with the
interest); switching the portfolio to `NOMINAL` turns "7% a year" into 7.229%
and lifts final wealth from €2.617m to €2.643m.

Yearly escalations — salary raises, rent increases, property appreciation — are
applied once a year and never converted to a monthly rate, so no convention
applies to them.

### Where the money goes

Every month the components produce signed cash flows in a fixed order — income,
rental income, spending, rent, debt service — and what is left over lands in the
**cash sink**: the `Etf` whose `monthly_investment` is `REMAINDER` (the
default). Exactly one portfolio must be the sink; give the others a fixed
`monthly_investment` amount. `Etf.cash(initial_capital=...)` is a zero-return,
untaxed portfolio, i.e. money in an account.

The sink also pays for the purchases at t=0 (down payment and fees) and covers
deficits by selling. Its capital may go **negative** when the cash flow cannot
fund the plan — that is an implicit loan at the portfolio's own rate, so check
`(df.etf_capital < 0).any()` before trusting a comparison.

### Sweeps

Models are immutable; `replaced()` returns a new one, so a sweep never disturbs
the baseline:

```python
model.replaced("home.mortgage", sondertilgung_yearly_rate=0.0)   # a component, or a nested object
model.replaced("etf", yearly_return_rate=0.05)
model.replaced(years=20)                                          # the model's own parameters
model.sweep("home.mortgage", "sondertilgung_yearly_rate", [0.0, 0.03, 0.05])
```

### The forecast

`run()` returns a DataFrame with one row per year (row 0 is the state right
after the t=0 purchases) and two kinds of columns:

- the model-wide totals — `total_loan`, `property_value`, `etf_capital`,
  `estimated_total_capital`, `monthly_leftover`, ... — each the sum over the
  components that feed it;
- a `<component>.<key>` block per component, e.g. `berlin.total_loan`,
  `pension.capital_after_tax`.

`flat(df)` drops the per-component block when you only want to look at the
totals.

Two of those totals are easy to confuse. `monthly_leftover` is the household
surplus **before** anything is withheld from it; `monthly_invested` is the part
that reached the portfolios. The gap is exactly `yearly_sondertilgung / 12` —
the money the principal took — so a plan can report a comfortable leftover
while the portfolio is being drained:

```
year  monthly_leftover  monthly_invested   # 5% Sondertilgung
   1           1192.98           -494.52
   2           1252.62           -434.88
```

Model semantics — year-0 row, tax and cost-basis treatment, the Sondertilgung
reserve, withdrawal ordering — are documented in the module docstring of
[src/model/\_\_init\_\_.py](src/model/__init__.py); the exact order in which the
engine calls the component hooks is in
[src/model/model.py](src/model/model.py).

## Notebooks

All notebooks build the same kind of model and differ only in components:

- `buy_vs_rent.ipynb` — buy your home with a mortgage vs rent + invest in ETF
- `sondertilgung_vs_etf.ipynb` — extra mortgage repayments vs ETF, breakeven ETF rate, early-repayment penalty, MSCI World stats
- `buy_as_investment_vs_etf.ipynb` — buy-to-let vs ETF-only
- `etf_withdrawal.ipynb` — ETF accumulation with a yearly percentage withdrawal

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

Three layers:

1. **Baseline equivalence** — `tests/golden/baseline_*.csv` pins the forecast of
   the pre-component model for every notebook configuration plus four edge
   cases. The component model must reproduce them **exactly**
   (`check_exact=True`), which is what keeps floating-point reassociation from
   silently moving the numbers.
2. **Legacy equivalence** — before that, the repo held three separate models;
   their outputs are pinned in `tests/golden/legacy_*.csv` and still reproduced
   wherever the semantics intentionally coincide. The deliberate deviations,
   including two money-leak bugs that were fixed, are documented in
   [tests/test_legacy_equivalence.py](tests/test_legacy_equivalence.py).
3. **Invariants** — cash conservation, closed-form annuity amortization and
   portfolio compounding, tax-on-gains-only, Sondertilgung caps, refinancing,
   validation, and that every total really is the sum of its components.

## Checking a change against the notebooks

```bash
.venv/bin/python scripts/run_notebooks.py --out artifacts/after
```

```bash
.venv/bin/python scripts/diff_notebook_outputs.py artifacts/baseline artifacts/after
```

The diff covers every number the notebooks print or render as a table; images
are excluded because they are not stable at the byte level.
