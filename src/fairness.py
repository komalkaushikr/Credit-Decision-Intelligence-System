"""Disparate-impact audit.

Compares approval rates across protected-style groupings (loan grade
and employment length) and flags any group whose approval rate deviates
more than a configurable threshold from the overall rate.

Note: in this dataset, *grade* is itself a risk factor that the lender
sets — so disparate impact across grades is expected. We still report
it because (a) interviewers ask, and (b) the magnitude is informative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ──> Standard rule-of-thumb thresholds.
DEVIATION_THRESHOLD = 0.20   # 20% deviation from overall approval rate
FOUR_FIFTHS_RATIO = 0.80     # 80% rule from the EEOC


def _approval_table(df: pd.DataFrame, group_col: str,
                    approve_col: str = "approve") -> pd.DataFrame:
    grouped = (df.groupby(group_col)
                 .agg(n=("default", "size"),
                      approval_rate=(approve_col, "mean"),
                      default_rate=("default", "mean"))
                 .reset_index())
    grouped = grouped[grouped["n"] >= 30]  # ignore tiny groups
    overall = float(df[approve_col].mean())
    grouped["overall_approval_rate"] = overall
    grouped["deviation"] = grouped["approval_rate"] - overall
    grouped["pct_deviation"] = grouped["deviation"] / overall if overall else np.nan
    grouped["four_fifths_ratio"] = grouped["approval_rate"] / overall if overall else np.nan
    grouped["flag_20pct"] = grouped["pct_deviation"].abs() > DEVIATION_THRESHOLD
    grouped["flag_four_fifths"] = grouped["four_fifths_ratio"] < FOUR_FIFTHS_RATIO
    return grouped.sort_values(group_col).reset_index(drop=True)


def audit_by_grade(df: pd.DataFrame, approve_col: str = "approve") -> pd.DataFrame:
    return _approval_table(df, "grade_letter", approve_col)


def audit_by_emp_length(df: pd.DataFrame, approve_col: str = "approve") -> pd.DataFrame:
    return _approval_table(df, "emp_length_raw", approve_col)


def summarise(audit_df: pd.DataFrame, group_col: str) -> dict:
    flagged = audit_df[audit_df["flag_20pct"]]
    return {
        "group_col": group_col,
        "n_groups": int(len(audit_df)),
        "overall_approval_rate": float(audit_df["overall_approval_rate"].iloc[0])
            if len(audit_df) else None,
        "n_flagged_20pct": int(len(flagged)),
        "flagged_groups": flagged[[group_col, "approval_rate",
                                   "pct_deviation"]].to_dict(orient="records"),
    }
