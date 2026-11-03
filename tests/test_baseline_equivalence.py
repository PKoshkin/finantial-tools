"""The component model must reproduce the pre-refactoring model bit for bit.

`tests/golden/baseline_*.csv` was produced by the old monolithic
`simulate(Scenario)` before it was replaced (see `scripts/capture_baseline.py`
in the commit that introduced those files). This suite rebuilds each of those
scenarios out of components and demands an *exact* match — `check_exact=True`,
not `pytest.approx`. Anything looser would hide the floating-point
reassociation traps the refactoring had to avoid (`a - b - c` vs `a - (b + c)`,
`L * r / 12` vs `L * (r / 12)`).

`tests/scenarios.py` is the component-side half of the contract: keep a builder
there for every pinned CSV.

One deliberate deviation, re-pinned rather than tolerated: the cash sink now
pays the gain tax on shares it sells to cover a deficit, which the old model
sold tax-free (and so did the component model at first). Only
`sondertilgung__4pct` and `sondertilgung__5pct` ever sell at a gain — their
Sondertilgung reserve outruns the leftover in years 1-3 — so only they were
re-pinned, and only in the five portfolio columns (`etf_capital`,
`etf_cost_basis`, `etf_capital_after_tax`, `estimated_total_capital`,
`monthly_costs_not_covered_by_safe_withdrawal`). The tax is 70 € and 175 €;
compounded, it costs 370 € and 903 € of final `estimated_total_capital`. The
same scenarios with an untaxed sink still reproduce the original fixtures'
`etf_capital` and `etf_cost_basis` bit for bit, and
`test_the_sale_tax_moves_only_the_portfolio` keeps every other column tied to
that untaxed run.
"""

from pathlib import Path

import pandas as pd
import pytest

from scenarios import BASELINE_MODELS
from src.model import FLAT_COLUMNS

GOLDEN = Path(__file__).parent / "golden"


def read_baseline(name: str) -> pd.DataFrame:
    """Load a pinned forecast, undoing the int downcast csv does on flat columns."""
    frame = pd.read_csv(GOLDEN / f"baseline_{name}.csv", float_precision="round_trip")
    return frame.astype({column: "float64" for column in frame.columns if column != "year"})


@pytest.mark.parametrize("name", sorted(BASELINE_MODELS))
def test_matches_baseline(name):
    expected = read_baseline(name)
    actual = BASELINE_MODELS[name]().run()

    # The component model adds per-component columns; the flat ones are the
    # contract with the old model.
    pd.testing.assert_frame_equal(
        actual[list(expected.columns)], expected, check_exact=True
    )


PORTFOLIO_COLUMNS = [
    "etf_capital",
    "etf_cost_basis",
    "etf_capital_after_tax",
    "estimated_total_capital",
    "monthly_costs_not_covered_by_safe_withdrawal",
]


@pytest.mark.parametrize("name", ["sondertilgung__4pct", "sondertilgung__5pct"])
def test_the_sale_tax_moves_only_the_portfolio(name):
    """The two re-pinned fixtures differ from an untaxed sink by the sale tax alone."""
    model = BASELINE_MODELS[name]()
    taxed = model.run()
    untaxed = model.replaced("etf", tax_rate=0.0).run()
    others = [column for column in FLAT_COLUMNS if column not in PORTFOLIO_COLUMNS]
    pd.testing.assert_frame_equal(taxed[others], untaxed[others], check_exact=True)
    paid = taxed["etf.yearly_sale_tax"].cumsum()
    assert paid.iloc[-1] > 0.0
    assert (taxed.etf_capital[paid > 0] < untaxed.etf_capital[paid > 0]).all()
    assert (taxed.etf_capital[paid == 0] == untaxed.etf_capital[paid == 0]).all()


@pytest.mark.parametrize("name", sorted(BASELINE_MODELS))
def test_flat_columns_are_complete(name):
    """No flat column may quietly disappear from the forecast."""
    actual = BASELINE_MODELS[name]().run()
    assert list(actual.columns[: 1 + len(FLAT_COLUMNS)]) == ["year", *FLAT_COLUMNS]


@pytest.mark.parametrize("name", sorted(BASELINE_MODELS))
def test_component_order_does_not_change_the_result(name):
    """The phase of a component decides when it is paid, not its list position."""
    model = BASELINE_MODELS[name]()
    reversed_model = type(model)(
        list(reversed(model.components)),
        years=model.years,
        start_age=model.start_age,
        yearly_inflation_rate=model.yearly_inflation_rate,
        safe_withdrawal_rate=model.safe_withdrawal_rate,
    )
    a = model.run()
    b = reversed_model.run()
    pd.testing.assert_frame_equal(a[list(FLAT_COLUMNS)], b[list(FLAT_COLUMNS)], check_exact=True)
