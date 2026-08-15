"""Turning per-component reports into the forecast DataFrame.

Every component reports its own unprefixed keys; this module writes them out as
`<component name>.<key>` and folds them into the flat, model-wide columns.

The rule for every flat column is "the sum of the per-component values that
feed it", so `df["home.total_loan"] + df["flat1.total_loan"] == df["total_loan"]`
holds by construction — including when there is exactly one contributor, which
is the case the pinned golden fixtures cover.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Columns whose value is the plain sum over the components of a given kind.
# (column, report_kind, key)
SUMMED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("total_loan", "property", "total_loan"),
    ("property_value", "property", "property_value"),
    ("property_equity", "property", "property_equity"),
    ("etf_capital", "etf", "capital"),
    ("etf_cost_basis", "etf", "cost_basis"),
    ("etf_capital_after_tax", "etf", "capital_after_tax"),
    ("monthly_income", "salary", "monthly_amount"),
    ("monthly_rental_income", "property", "monthly_rental_income"),
    ("monthly_spending", "spending", "monthly_amount"),
    ("monthly_rent", "rent", "monthly_amount"),
    ("monthly_interest_payment", "property", "monthly_interest_payment"),
    ("monthly_loan_repayment", "property", "monthly_loan_repayment"),
    ("monthly_mortgage_payment", "property", "monthly_mortgage_payment"),
    ("yearly_sondertilgung", "property", "yearly_sondertilgung"),
    ("cumulative_interest_paid", "property", "cumulative_interest_paid"),
    ("yearly_etf_withdrawal_gross", "etf", "yearly_withdrawal_gross"),
    ("yearly_etf_withdrawal_net", "etf", "yearly_withdrawal_net"),
)

# The flat columns, in the order they appear in the DataFrame.
FLAT_COLUMNS: tuple[str, ...] = (
    "total_loan",
    "property_value",
    "property_equity",
    "etf_capital",
    "etf_cost_basis",
    "etf_capital_after_tax",
    "estimated_total_capital",
    "invested_capital",
    "monthly_income",
    "monthly_rental_income",
    "monthly_spending",
    "monthly_rent",
    "monthly_interest_payment",
    "monthly_loan_repayment",
    "monthly_mortgage_payment",
    "monthly_housing_cost",
    "monthly_leftover",
    "monthly_invested",
    "yearly_sondertilgung",
    "cumulative_interest_paid",
    "yearly_etf_withdrawal_gross",
    "yearly_etf_withdrawal_net",
    "monthly_costs_not_covered_by_safe_withdrawal",
)

RESERVED_NAMES = frozenset(FLAT_COLUMNS) | {"year", "age"}


@dataclass(frozen=True, slots=True)
class ReportSource:
    """One component's contribution to a row."""

    name: str
    kind: str
    values: dict[str, float]


def build_row(
    *,
    year: int,
    start_age: int | None,
    sources: list[ReportSource],
    invested_capital: float,
    monthly_leftover: float | None,
    monthly_invested: float | None,
    safe_withdrawal_rate: float,
) -> dict[str, float]:
    """Assemble one forecast row.

    `monthly_leftover=None` marks the year-0 row, where the leftover is derived
    from the components' initial values rather than averaged over 12 months.

    `monthly_leftover` is the household's surplus *before* anything is withheld
    from it; `monthly_invested` is the part that reached the portfolios. The
    gap is exactly `yearly_sondertilgung / 12` — the money the principal took.
    A reserve that is withheld but that the loan cannot absorb is handed back
    to the sink at year end, so it counts as invested, which is why the gap
    tracks what was *applied* rather than what was withheld. At year 0 no month
    has run yet and the two are equal.
    """
    row: dict[str, float] = {"year": year}
    if start_age is not None:
        row["age"] = start_age + year

    for column, kind, key in SUMMED_COLUMNS:
        total = 0.0
        for source in sources:
            if source.kind == kind:
                total += source.values[key]
        row[column] = total

    row["estimated_total_capital"] = row["property_equity"] + row["etf_capital_after_tax"]
    row["monthly_housing_cost"] = row["monthly_rent"] + row["monthly_mortgage_payment"]
    row["invested_capital"] = invested_capital
    if monthly_leftover is None:
        # Operating costs have no flat column (only rental flats report them,
        # so a summed column would break the shared "property" kind), hence
        # the `.get` rather than a SUMMED_COLUMNS entry.
        operating_costs = 0.0
        for source in sources:
            if source.kind == "property":
                operating_costs += source.values.get("monthly_operating_costs", 0.0)
        row["monthly_leftover"] = (
            row["monthly_income"]
            + row["monthly_rental_income"]
            - row["monthly_spending"]
            - row["monthly_housing_cost"]
            - operating_costs
        )
    else:
        row["monthly_leftover"] = monthly_leftover
    row["monthly_invested"] = (
        row["monthly_leftover"] if monthly_invested is None else monthly_invested
    )
    row["monthly_costs_not_covered_by_safe_withdrawal"] = (
        row["monthly_spending"] + row["monthly_housing_cost"]
    ) - row["etf_capital_after_tax"] * safe_withdrawal_rate / 12.0

    for source in sources:
        for key, value in source.values.items():
            row[f"{source.name}.{key}"] = value
    return row


def to_frame(rows: list[dict[str, float]], *, start_age: int | None) -> pd.DataFrame:
    """Build the DataFrame with a stable column order and stable dtypes."""
    head = ["year"] + (["age"] if start_age is not None else [])
    detail = [key for key in rows[0] if key not in head and key not in FLAT_COLUMNS]
    frame = pd.DataFrame(rows, columns=head + list(FLAT_COLUMNS) + detail)
    return frame.astype({column: "float64" for column in frame.columns if column not in head})


def flat(forecast: pd.DataFrame) -> pd.DataFrame:
    """The model-wide columns only, without the per-component detail.

    `FinancialModel.run()` returns both: the flat totals and a
    `<component>.<key>` block per component. The per-component block is what
    you drill into; this is what you look at.
    """
    head = [column for column in ("year", "age") if column in forecast.columns]
    return forecast[head + [column for column in FLAT_COLUMNS if column in forecast.columns]]


def payoff_year(model: pd.DataFrame) -> int | None:
    """First year (>= 1) in which every mortgage is repaid, or None."""
    paid_off = model[(model["year"] > 0) & (model["total_loan"] <= 0)]
    return int(paid_off["year"].iloc[0]) if not paid_off.empty else None


def calculate_early_repayment_penalty(
    remaining_principal: float,
    loan_rate: float,
    market_rate: float,
    remaining_years: float,
) -> float:
    """Approximate Vorfälligkeitsentschädigung (early repayment penalty).

    After the Zinsbindung ends the penalty is 0 (with 6 months notice). Before
    that, it compensates the bank for the interest rate differential over the
    remaining fixed-rate period.
    """
    if remaining_years <= 0:
        return 0.0
    return remaining_principal * max(0.0, loan_rate - market_rate) * remaining_years
