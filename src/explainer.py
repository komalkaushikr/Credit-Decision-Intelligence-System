"""SHAP utilities for the PD model.

Falls back to sklearn feature importances if `shap` isn't installed,
so the rest of the pipeline still runs in minimal environments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    import shap  # type: ignore
    _SHAP_OK = True
except Exception:  # pragma: no cover
    shap = None
    _SHAP_OK = False


# Plain-English glosses for the most common LendingClub features.
FEATURE_GLOSS = {
    "fico": "credit-bureau FICO score — higher means more reliable past repayment",
    "int_rate": "interest rate the loan is priced at — higher rates flag riskier borrowers",
    "grade_num": "LendingClub risk grade (A→G) — lower is safer",
    "subgrade_num": "fine-grained risk grade (A1→G5)",
    "dti": "debt-to-income ratio — how stretched the borrower already is",
    "annual_inc": "borrower's annual income",
    "loan_to_income": "loan size relative to income — how much capacity is being consumed",
    "payment_to_income": "monthly payment as a share of monthly income",
    "revol_util": "% of revolving credit currently used — a strain indicator",
    "credit_age_yrs": "length of credit history — older files are typically safer",
    "delinq_2yrs": "delinquencies in the last 2 years",
    "inq_last_6mths": "credit inquiries in the last 6 months — many inquiries = credit-hungry",
    "open_acc": "number of currently open credit lines",
    "pub_rec": "public derogatory records (bankruptcies etc.)",
    "term_months": "loan term in months",
    "emp_length_num": "years at current employer",
    "home_ownership": "rent / own / mortgage status",
    "purpose": "what the borrower says the loan is for",
}


@dataclass
class SHAPBundle:
    values: np.ndarray             # (n_samples, n_features)
    expected_value: float
    feature_names: list
    X: pd.DataFrame                # the input frame the values correspond to


def _unwrap_calibrated(estimator):
    """Pull the underlying tree estimator out of a CalibratedClassifierCV."""
    inner = estimator
    if hasattr(estimator, "calibrated_classifiers_"):
        inner = estimator.calibrated_classifiers_[0].estimator
    return inner


def compute_shap(model, X: pd.DataFrame, max_samples: int = 2000) -> Optional[SHAPBundle]:
    """Returns a SHAPBundle, or None if `shap` isn't available."""
    if not _SHAP_OK:
        return None
    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=42)
    inner = _unwrap_calibrated(model.estimator)
    try:
        explainer = shap.TreeExplainer(inner)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        expected = explainer.expected_value
        if isinstance(expected, (list, np.ndarray)) and np.ndim(expected) > 0:
            expected = float(np.asarray(expected).ravel()[-1])
        return SHAPBundle(values=np.asarray(sv), expected_value=float(expected),
                          feature_names=list(X.columns), X=X.reset_index(drop=True))
    except Exception:
        return None


def global_importance(bundle: SHAPBundle) -> pd.DataFrame:
    mean_abs = np.abs(bundle.values).mean(axis=0)
    df = pd.DataFrame({"feature": bundle.feature_names,
                       "mean_abs_shap": mean_abs})
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def top_features_plain_english(bundle: SHAPBundle, k: int = 3) -> list[dict]:
    """Top-k features with business-language interpretation."""
    imp = global_importance(bundle).head(k)
    out = []
    for _, row in imp.iterrows():
        feat = row["feature"]
        out.append({
            "feature": feat,
            "mean_abs_shap": float(row["mean_abs_shap"]),
            "interpretation": FEATURE_GLOSS.get(
                feat, f"{feat} — quantitative input to the risk score"),
        })
    return out


def fallback_importance(model, feature_cols: list[str]) -> pd.DataFrame:
    """Sklearn feature_importances_ when SHAP isn't installed."""
    inner = _unwrap_calibrated(model.estimator)
    if not hasattr(inner, "feature_importances_"):
        return pd.DataFrame({"feature": feature_cols,
                             "importance": np.nan})
    return (pd.DataFrame({"feature": feature_cols,
                          "importance": inner.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True))
