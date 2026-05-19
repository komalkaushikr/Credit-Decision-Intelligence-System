"""Central configuration: paths, business constants, columns.

All other modules import from here. Override CSV location via env var
CREDIT_DATA_PATH so the same code runs on any machine.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Default points at the user's local LendingClub CSV; override via env var.
_DEFAULT_CSV = r"C:\Users\komal\Desktop\e-e business analytics\accepted_2007_to_2018q4.csv\accepted_2007_to_2018Q4.csv"
RAW_CSV_PATH = Path(os.environ.get("CREDIT_DATA_PATH", _DEFAULT_CSV))

# How many rows to load. None = full file. Keep capped during development.
N_ROWS = int(os.environ.get("CREDIT_N_ROWS", "300000"))

# ─── Loan economics ──────────────────────────────────────────────────────────
LGD = 0.80          # Loss Given Default — fraction of principal lost
RECOVERY = 0.20     # Fraction recovered on default
OPEX_RATE = 0.01    # Origination + servicing cost as fraction of principal
COST_OF_CAP = 0.04  # Opportunity cost of capital (annualised)

# Hard underwriting floors used by the production engine.
MIN_FICO = 600
MAX_DTI = 40.0
MAX_PD_OVERRIDE = 0.55  # never approve above this PD regardless of profit

# ─── Columns ────────────────────────────────────────────────────────────────
# Underwriting-time columns + the dates needed for survival & walk-forward.
KEEP_COLS = [
    "loan_status", "loan_amnt", "term", "int_rate", "installment",
    "grade", "sub_grade", "purpose", "application_type",
    "annual_inc", "verification_status", "dti", "emp_length",
    "home_ownership", "addr_state",
    "fico_range_low", "fico_range_high", "earliest_cr_line",
    "open_acc", "total_acc", "revol_bal", "revol_util",
    "pub_rec", "delinq_2yrs", "inq_last_6mths", "mort_acc",
    "tot_cur_bal", "tot_hi_cred_lim",
    # Dates added for survival analysis & walk-forward backtest.
    "issue_d", "last_pymnt_d",
]

FINANCIAL_COLS = ["loan_amnt", "int_rate", "term_months", "installment"]

# Output artifact paths.
MODEL_PATH = OUTPUTS_DIR / "model.pkl"
THRESHOLD_PATH = OUTPUTS_DIR / "threshold.json"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"
FEATURE_COLS_PATH = OUTPUTS_DIR / "feature_cols.json"
CLEAN_PARQUET_PATH = OUTPUTS_DIR / "clean.parquet"
