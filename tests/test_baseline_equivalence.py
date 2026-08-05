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
