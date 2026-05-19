"""Temporal walk-forward backtesting.

Train the PD model on loans issued before a cutoff year; evaluate on each
subsequent year separately. Reports profit per 1,000 loans for each test
year — proves the model's edge isn't a snapshot artifact.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from . import metrics as M


def walk_forward_backtest(df: pd.DataFrame, feature_cols: list[str],
                          train_until_year: int,
                          test_years: list[int] | None = None,
                          seed: int = 42) -> pd.DataFrame:
    """For each test year, train on everything strictly before that year,
    score, apply the profit-optimal policy, report results."""
    df = df.dropna(subset=["issue_year"]).copy()
    df["issue_year"] = df["issue_year"].astype(int)

    if test_years is None:
        max_year = int(df["issue_year"].max())
        test_years = list(range(train_until_year + 1, max_year + 1))

    rows = []
    for test_year in test_years:
        train_mask = df["issue_year"] <= train_until_year
        test_mask = df["issue_year"] == test_year
        if train_mask.sum() < 1000 or test_mask.sum() < 100:
            continue

        # Retrain on the expanding window so each test year uses everything
        # known up to (and including) the cutoff — classic walk-forward.
        train_df = df[train_mask]
        test_df = df[test_mask].copy()
        X_train = train_df[feature_cols].astype(float).fillna(0.0)
        y_train = train_df["default"].astype(int)
        X_test = test_df[feature_cols].astype(float).fillna(0.0)

        clf = GradientBoostingClassifier(
            n_estimators=120, max_depth=4, learning_rate=0.08,
            subsample=0.9, random_state=seed,
        )
        clf.fit(X_train, y_train)
        test_df["pd"] = clf.predict_proba(X_test)[:, 1]

        stats = M.compare_to_naive(test_df)
        rows.append({
            "test_year": test_year,
            "n_test": int(test_mask.sum()),
            "default_rate": float(test_df["default"].mean()),
            "approval_rate": stats["optimized_approval_rate"],
            "profit_per_1000": stats["optimized_profit_per_1000"],
            "naive_profit_per_1000": stats["naive_profit_per_1000"],
            "extra_profit_per_1000": stats["extra_profit_per_1000"],
        })
    return pd.DataFrame(rows)


def profit_retention(walk_df: pd.DataFrame) -> dict:
    """How much of year-1's profit/1000 do we retain in later test years?"""
    if walk_df.empty:
        return {}
    first = walk_df.iloc[0]
    last = walk_df.iloc[-1]
    base = first["profit_per_1000"]
    retention = (last["profit_per_1000"] / base) if base != 0 else float("nan")
    return {
        "first_year": int(first["test_year"]),
        "last_year": int(last["test_year"]),
        "first_profit_per_1000": float(base),
        "last_profit_per_1000": float(last["profit_per_1000"]),
        "retention_pct": float(retention),
        "mean_profit_per_1000": float(walk_df["profit_per_1000"].mean()),
        "std_profit_per_1000": float(walk_df["profit_per_1000"].std()),
    }
