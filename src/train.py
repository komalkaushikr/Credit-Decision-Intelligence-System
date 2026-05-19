"""End-to-end training script.

Run once locally:
    python -m src.train

Produces in outputs/:
  - model.pkl                 — pickled TrainedModel
  - feature_cols.json         — feature contract
  - threshold.json            — profit-optimal fixed PD threshold
  - metrics.json              — AUC + headline profit numbers
  - clean.parquet             — modeling-ready frame (for dashboard batch demo)
"""
from __future__ import annotations

import json

from . import config
from . import data_loader
from . import metrics as M
from . import model as Mdl


def main() -> None:
    print(f"Loading data from {config.RAW_CSV_PATH} (nrows={config.N_ROWS}) …")
    df = data_loader.load_clean()
    print(f"Clean frame: {len(df):,} rows × {df.shape[1]} cols, "
          f"default rate {df['default'].mean():.2%}")

    feature_cols = data_loader.feature_columns(df)
    print(f"Training on {len(feature_cols)} features")

    trained, model_metrics = Mdl.train_pd_model(df, feature_cols)
    print(f"AUC test: {model_metrics['auc_test']:.3f}")

    # Score the *whole* frame so the dashboard can demo on real rows.
    df["pd"] = trained.predict_pd(df)

    headline = M.compare_to_naive(df)
    print(M.headline_summary(headline))

    best_fixed = M.find_optimal_fixed_threshold(df)
    print(f"Optimal fixed PD threshold: {best_fixed['threshold']:.3f} "
          f"→ profit ${best_fixed['profit']:,.0f}")

    # ─── Persist artifacts ──────────────────────────────────────────────
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    Mdl.save_model(trained)
    Mdl.save_json(feature_cols, config.FEATURE_COLS_PATH)
    Mdl.save_json({"fixed_threshold": best_fixed,
                   "policy": "approve iff expected_profit > 0 "
                   "(per-loan break-even)"},
                  config.THRESHOLD_PATH)
    Mdl.save_json({"model": model_metrics, "policy": headline},
                  config.METRICS_PATH)

    # Save a slim parquet for the dashboard batch demo.
    keep = (["loan_amnt", "int_rate", "term_months", "fico", "dti",
             "annual_inc", "purpose", "grade_letter", "emp_length_raw",
             "default", "pd"] + feature_cols)
    keep = list(dict.fromkeys(keep))  # dedupe, preserve order
    try:
        df[keep].sample(min(20000, len(df)), random_state=42)\
               .to_parquet(config.CLEAN_PARQUET_PATH, index=False)
        print(f"Wrote sample parquet → {config.CLEAN_PARQUET_PATH}")
    except Exception as exc:  # pyarrow optional
        print(f"Skipped parquet ({exc}); saving CSV instead.")
        df[keep].sample(min(20000, len(df)), random_state=42)\
               .to_csv(config.CLEAN_PARQUET_PATH.with_suffix(".csv"),
                       index=False)

    print("\nArtifacts written to outputs/. Now run:")
    print("    streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
