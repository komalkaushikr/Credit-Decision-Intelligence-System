"""Survival analysis: how long until a loan defaults?

- Kaplan-Meier curves segmented by grade A-G.
- Cox Proportional Hazards model for time-to-default prediction.
- Median survival times and pairwise differences across grades.

Requires `lifelines`. If unavailable we raise a clear error so the notebook
can surface a "pip install lifelines" message instead of silently failing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter  # type: ignore
    from lifelines.statistics import logrank_test         # type: ignore
    _LL_OK = True
except Exception:  # pragma: no cover
    KaplanMeierFitter = None
    CoxPHFitter = None
    logrank_test = None
    _LL_OK = False


def _require_lifelines() -> None:
    if not _LL_OK:
        raise ImportError(
            "lifelines is not installed. Run `pip install lifelines` "
            "to use the survival module."
        )


@dataclass
class KMResult:
    grade: str
    median_months: float
    survival_at_12mo: float
    survival_at_36mo: float
    n: int
    n_events: int


def km_by_grade(df: pd.DataFrame) -> tuple[dict[str, "KaplanMeierFitter"], pd.DataFrame]:
    """Fit one KM curve per grade. Returns the fitters and a summary table."""
    _require_lifelines()
    fitters: dict[str, KaplanMeierFitter] = {}
    rows: list[KMResult] = []
    for grade in sorted(df["grade_letter"].dropna().unique()):
        sub = df[df["grade_letter"] == grade]
        if len(sub) < 50:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(durations=sub["duration_months"], event_observed=sub["event"],
                label=f"Grade {grade}")
        fitters[grade] = kmf

        median = float(kmf.median_survival_time_) if not np.isnan(
            kmf.median_survival_time_) else float("inf")
        # survival probabilities at fixed horizons
        s12 = float(kmf.survival_function_at_times(12).iloc[0])
        s36 = float(kmf.survival_function_at_times(36).iloc[0])
        rows.append(KMResult(grade=grade, median_months=median,
                             survival_at_12mo=s12, survival_at_36mo=s36,
                             n=len(sub), n_events=int(sub["event"].sum())))
    summary = pd.DataFrame([r.__dict__ for r in rows]).sort_values("grade")
    return fitters, summary.reset_index(drop=True)


def pairwise_logrank(df: pd.DataFrame) -> pd.DataFrame:
    """Logrank p-values between consecutive grades — proves the curves differ."""
    _require_lifelines()
    grades = sorted(df["grade_letter"].dropna().unique())
    rows = []
    for a, b in zip(grades[:-1], grades[1:]):
        A = df[df["grade_letter"] == a]
        B = df[df["grade_letter"] == b]
        res = logrank_test(A["duration_months"], B["duration_months"],
                           A["event"], B["event"])
        rows.append({"grade_a": a, "grade_b": b,
                     "test_statistic": float(res.test_statistic),
                     "p_value": float(res.p_value)})
    return pd.DataFrame(rows)


def fit_cox(df: pd.DataFrame, covariates: list[str],
            duration_col: str = "duration_months",
            event_col: str = "event") -> "CoxPHFitter":
    """Cox PH fit on a small, interpretable covariate set."""
    _require_lifelines()
    cols = covariates + [duration_col, event_col]
    sub = df[cols].dropna().copy()
    # Drop near-constant columns; Cox will fail otherwise.
    sub = sub.loc[:, sub.nunique() > 1]
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(sub, duration_col=duration_col, event_col=event_col, show_progress=False)
    return cph


def median_survival_gap(summary: pd.DataFrame) -> dict:
    """Difference in median survival between the safest and riskiest grade present."""
    finite = summary[np.isfinite(summary["median_months"])]
    if finite.empty:
        return {"note": "No grade reached its median — all curves stay above 0.5"}
    best = finite.iloc[0]
    worst = finite.iloc[-1]
    return {
        "safest_grade": str(best["grade"]),
        "safest_median_months": float(best["median_months"]),
        "riskiest_grade": str(worst["grade"]),
        "riskiest_median_months": float(worst["median_months"]),
        "gap_months": float(best["median_months"] - worst["median_months"]),
    }
