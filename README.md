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
| `GermanHousehold` — gross salaries under German taxation | 0 or 1 | `earners` (1-2 × `Earner`), `joint`, `tax_classes`, `n_children`, `tax` (a `TaxSystem`) |
| `Spending` | any | `monthly_amount`, `yearly_increase_rate` (`None` → the model's inflation), `compounding` |
| `Rent` — your own housing | 0 or 1 | `monthly_amount`, `yearly_increase_rate` |
| `Home` — the place you live in | 0 or 1 | `price`, `yearly_price_increase_rate`, `mortgage` |
| `InvestmentProperty` — bought to rent out | any | the above plus `monthly_rental_income`, `yearly_rental_income_increase_rate`, `operating_costs`, `vacancy_rate`, `taxation` |
| `Etf` — a portfolio | any | `initial_capital`, `yearly_return_rate`, `tax_rate`, `monthly_investment`, `yearly_withdrawal_rate`, `compounding` |

Every component also takes a `name`, which prefixes its own result columns.

`Mortgage` is not a component but a value object nested in a property
(`interest_rate`, `down_payment_rate`, `fees_rate`, `yearly_repayment_rate`,
`refinancing_years`, `sondertilgung_yearly_rate`, `compounding`), because a loan
here always belongs to one property and is derived from its price. A rental
flat nests two more value objects the same way: `OperatingCosts` and
`RentalTaxation` (see below). An `Earner` may nest a `PrivateHealthInsurance`
(`monthly_premium`, `monthly_care_premium`, `deductible_share`,
`yearly_increase_rate`, `monthly_rebate` — a Pauschalerstattung netted against
the debit, cash-only: payroll and assessment keep using the full contract
figures). The payroll chain is pinned against a real 2026 DATEV payslip in
`tests/test_real_payslips.py`.

Rates of return are per component: two portfolios can grow at different rates
and be taxed differently.

### Rental taxation

A bare `InvestmentProperty` treats its rent as tax-free cash. Three optional
parameters model a German buy-to-let properly:

```python
InvestmentProperty(
    price=510_000,
    monthly_rental_income=2_820,            # Nettokaltmiete incl. Stellplatz
    yearly_rental_income_increase_rate=0.02,
    yearly_price_increase_rate=0.02,
    mortgage=Mortgage(interest_rate=0.045, down_payment_rate=0.0,
                      fees_rate=0.12, yearly_repayment_rate=0.015),
    operating_costs=OperatingCosts(
        monthly_deductible=150.0,           # non-recoverable Hausgeld, Grundsteuer, ...
        monthly_own_reserve=206.0,          # own maintenance reserve: cash, not deductible
        yearly_increase_rate=0.02,
    ),
    vacancy_rate=0.0,                       # share of rent lost to vacancy
    taxation=RentalTaxation(afa_rate=0.03, building_share=0.96),
)
```

With `taxation` set the flat publishes its yearly Einkünfte aus Vermietung und
Verpachtung — rent minus deductible costs, loan interest and AfA — to the
model's `GermanHousehold` (required then; the model refuses to build without
one). The household re-assesses its income tax with that result in the zvE, so
a rental loss becomes a refund and a profit a back-payment, arriving with the
July settlement of the *following* year like any other refund. The AfA base is
`(price + purchase fees) * building_share`, depreciating at `afa_rate` a year.

Deliberate simplifications: umlagefähige Kosten are not modelled (they cancel
between the Warmmiete and the expenses — the rent here is a Nettokaltmiete);
a vacancy reduces the collected rent *and* the taxable result, unlike planning
calculators that book a kalkulatorischer Mietausfall on top of full rent; AfA
runs flat forever, overstating the deduction beyond `1 / afa_rate` years;
losses only offset income within their year (no Verlustvortrag); and the final
year's tax settlement falls beyond the horizon and is dropped — conservative
for a refund, but for a profitable flat it means the last year's rental tax is
never charged.

### One year, without a forecast

Not every question needs a simulation. `src/model/german_tax.py` holds the
German rules as pure functions over plain numbers, and `employee_year` runs the
whole chain for a single calendar year — from what an employee costs the
employer down to what the year is finally worth:

```python
from src.model.german_tax import TaxSystem, employee_year

year = employee_year(100_000, TaxSystem())     # single, childless, statutory
year.employer_cost                             # 117,958.63
year.net                                       #  58,014.36
year.take_home_share                           #       0.4918
```

Married, with children, privately insured or on another Steuerklasse are
keyword arguments; `GermanHousehold` runs the same arithmetic inside the engine
and the two are pinned against each other in the tests. Two cautions. The
employer cost covers the four social insurance branches only — the Umlagen
U1/U2, the Insolvenzgeldumlage and the accident insurance are not modelled, so
a real employer pays another one to two percent of the gross. And `net` is what
the year is worth once the tax return has settled, while `net_paid_out` is the
twelve payslips; `refund` is the difference between them.

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
- `take_home_share.ipynb` — net pay as a share of total employer cost, across the German salary range

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
