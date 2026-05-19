"""Load + clean + feature-engineer LendingClub accepted-loans data.

Single source of truth used by the training script, the analysis notebooks,
and the Streamlit dashboard. Mirrors Phase 1 of the original notebook but
also keeps `issue_d` and `last_pymnt_d` so we can do survival analysis and
temporal walk-forward backtesting.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from . import config

AMBIGUOUS_STATUS = [
    "Current", "In Grace Period", "Late (16-30 days)",
    "Late (31-120 days)", "Issued",
]
DEFAULT_STATUS = ["Charged Off", "Default"]

EMP_MAP = {
    "< 1 year": 0.5, "1 year": 1, "2 years": 2, "3 years": 3,
    "4 years": 4, "5 years": 5, "6 years": 6, "7 years": 7,
    "8 years": 8, "9 years": 9, "10+ years": 11, "n/a": -1,
}
GRADE_MAP = {g: i for i, g in enumerate("ABCDEFG")}
SUBGRADE_MAP = {f"{g}{n}": i for i, (g, n) in
                enumerate([(g, n) for g in "ABCDEFG" for n in range(1, 6)])}


def load_raw(csv_path: Path | None = None, nrows: int | None = None) -> pd.DataFrame:
    csv_path = Path(csv_path or config.RAW_CSV_PATH)
    nrows = nrows if nrows is not None else config.N_ROWS
    if not csv_path.exists():
        raise FileNotFoundError(
            f"LendingClub CSV not found at {csv_path}. "
            f"Set the CREDIT_DATA_PATH env var to point at your local copy."
        )
    return pd.read_csv(csv_path, usecols=config.KEEP_COLS,
                       low_memory=False, nrows=nrows)


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Binary target.
    df["default"] = df["loan_status"].isin(DEFAULT_STATUS).astype(int)

    # Keep raw grade letter for fairness/survival groupings BEFORE encoding.
    df["grade_letter"] = df["grade"].astype(str)
    df["emp_length_raw"] = df["emp_length"].fillna("Unknown").astype(str)

    # term → integer months.
    df["term_months"] = df["term"].str.extract(r"(\d+)").astype(float)

    # FICO midpoint.
    df["fico"] = (df["fico_range_low"] + df["fico_range_high"]) / 2

    # Dates.
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df["last_pymnt_d"] = pd.to_datetime(df["last_pymnt_d"], format="%b-%Y", errors="coerce")
    df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"],
                                            format="%b-%Y", errors="coerce")
    ref_date = pd.Timestamp("2018-12-31")
    df["credit_age_yrs"] = (ref_date - df["earliest_cr_line"]).dt.days / 365.25
    df["issue_year"] = df["issue_d"].dt.year

    # Time-to-event for survival (months from issue to last payment / term end).
    months_observed = ((df["last_pymnt_d"] - df["issue_d"]).dt.days / 30.44)
    df["duration_months"] = months_observed.clip(lower=0).fillna(df["term_months"])
    # event = defaulted within the observation window.
    df["event"] = df["default"]

    # Employment length numeric.
    df["emp_length_num"] = df["emp_length"].map(EMP_MAP).fillna(-1)

    # Derived ratios.
    df["monthly_inc"] = df["annual_inc"] / 12
    df["payment_to_income"] = df["installment"] / df["monthly_inc"].clip(lower=1)
    df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"].clip(lower=1)
    df["debt_to_credit"] = df["tot_cur_bal"] / df["tot_hi_cred_lim"].clip(lower=1)
    df["acc_breadth"] = df["open_acc"] / df["total_acc"].clip(lower=1)
    df["high_util_flag"] = (df["revol_util"] > 80).astype(int)

    # Ordinal grade encodings.
    df["grade_num"] = df["grade"].map(GRADE_MAP).fillna(6).astype(int)
    df["subgrade_num"] = df["sub_grade"].map(SUBGRADE_MAP).fillna(34).astype(int)

    # Frequency-encode state.
    state_freq = df["addr_state"].value_counts(normalize=True)
    df["state_freq"] = df["addr_state"].map(state_freq)

    # Label-encode remaining low-cardinality categoricals.
    for col in ["purpose", "application_type", "verification_status", "home_ownership"]:
        df[col] = df[col].fillna("Unknown").astype(str)
        df[col] = LabelEncoder().fit_transform(df[col])

    # Median-impute numeric NaNs.
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    return df


def load_clean() -> pd.DataFrame:
    """Return modeling-ready dataframe. Use this everywhere."""
    raw = load_raw()
    raw = raw[~raw["loan_status"].isin(AMBIGUOUS_STATUS)].copy()
    df = _engineer(raw)
    # Drop rows still missing the target or issue date — they're useless.
    df = df.dropna(subset=["default", "issue_d"])
    df = df.reset_index(drop=True)
    return df


# Columns the model trains on. Excludes financial/leaky/grouping cols.
EXCLUDED_FROM_FEATURES = {
    "default", "event", "duration_months",
    "loan_status", "issue_d", "last_pymnt_d", "earliest_cr_line",
    "term", "grade", "sub_grade", "addr_state", "emp_length",
    "grade_letter", "emp_length_raw", "issue_year", "monthly_inc",
    *config.FINANCIAL_COLS,
}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in EXCLUDED_FROM_FEATURES
            and pd.api.types.is_numeric_dtype(df[c])]
