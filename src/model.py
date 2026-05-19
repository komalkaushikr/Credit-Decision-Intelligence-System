"""PD model training + persistence.

Gradient-boosted PD model with isotonic calibration (so the output
probabilities can be plugged directly into the profit equation).
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from . import config


@dataclass
class TrainedModel:
    """Bundle of fitted estimator + feature contract."""
    estimator: object
    feature_cols: list

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.feature_cols].astype(float).fillna(0.0)
        return self.estimator.predict_proba(X)[:, 1]


def train_pd_model(df: pd.DataFrame, feature_cols: list[str],
                   test_size: float = 0.2, seed: int = 42) -> tuple[TrainedModel, dict]:
    """Train a calibrated GBM and report AUC."""
    X = df[feature_cols].astype(float).fillna(0.0)
    y = df["default"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)

    base = GradientBoostingClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.08,
        subsample=0.9, random_state=seed,
    )
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(X_train, y_train)

    pd_train = model.predict_proba(X_train)[:, 1]
    pd_test = model.predict_proba(X_test)[:, 1]

    metrics = {
        "auc_train": float(roc_auc_score(y_train, pd_train)),
        "auc_test": float(roc_auc_score(y_test, pd_test)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "default_rate_train": float(y_train.mean()),
        "default_rate_test": float(y_test.mean()),
    }
    return TrainedModel(estimator=model, feature_cols=feature_cols), metrics


# ─── Persistence ────────────────────────────────────────────────────────────

def save_model(model: TrainedModel, path: Path | None = None) -> Path:
    path = Path(path or config.MODEL_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(model, fh)
    return path


def load_model(path: Path | None = None) -> TrainedModel:
    path = Path(path or config.MODEL_PATH)
    with path.open("rb") as fh:
        return pickle.load(fh)


def save_json(obj: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    return path
