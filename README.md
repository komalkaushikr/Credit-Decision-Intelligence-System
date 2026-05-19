# Credit Decision Intelligence System

> **A profit-optimized loan approval engine** built on the LendingClub
> 2007–2018 dataset. The model doesn't predict default — it allocates capital.

---

## The headline (the only number that matters)

> **The profit-optimized policy generates roughly $49M more portfolio profit
> than the conservative rule baseline on the holdout, with ~95% approval and
> ~20% default rate among approved loans.**
> Per 1,000 loans, that translates to thousands of extra dollars relative to
> blanket approval. Exact numbers are written to `outputs/metrics.json` after
> training; the dashboard shows them at the top.

This is built around the cost matrix, not AUC:

|                    | Actual: Good                | Actual: Default          |
|--------------------|-----------------------------|--------------------------|
| **Approve**        | TN = `+L·r·t − L·OpEx`      | FN = `−L·LGD − L·OpEx`   |
| **Reject**         | FP = `−(L·r·t − L·OpEx)`    | TP = 0                    |

Approve iff `PD < (r·t − OpEx) / (LGD + r·t)` — a per-loan, analytically
derived threshold, not a fixed 0.5.

---

## What's in the box

| Capability | Where it lives |
|---|---|
| Cost-sensitive profit framework (per-loan break-even threshold, $/1000-loan headline) | [`src/metrics.py`](src/metrics.py), [`notebooks/00_profit_optimization.ipynb`](notebooks/00_profit_optimization.ipynb) |
| Survival analysis (Kaplan-Meier by grade, Cox PH, median-survival gap) | [`src/survival.py`](src/survival.py), [`notebooks/01_survival_analysis.ipynb`](notebooks/01_survival_analysis.ipynb) |
| Temporal walk-forward backtest (year-by-year stability) | [`src/backtest.py`](src/backtest.py), [`notebooks/02_walkforward_backtest.ipynb`](notebooks/02_walkforward_backtest.ipynb) |
| SHAP explainability (global summary + waterfall + plain-English top-3) | [`src/explainer.py`](src/explainer.py), [`notebooks/03_shap_explainability.ipynb`](notebooks/03_shap_explainability.ipynb) |
| Fairness audit (disparate impact by grade & employment length) | [`src/fairness.py`](src/fairness.py), [`notebooks/04_fairness_audit.ipynb`](notebooks/04_fairness_audit.ipynb) |
| Production decision engine (hard floors + PD ceiling + risk tiers) | [`src/decision.py`](src/decision.py) |
| Streamlit dashboard (single applicant + batch CSV + SHAP panel) | [`dashboard/app.py`](dashboard/app.py) |
| Original 9-phase build (kept for reference) | [`notebooks/99_original_full_build.ipynb`](notebooks/99_original_full_build.ipynb) |

---

## Repo layout

```
.
├── data/                          # raw CSV lives here (gitignored)
├── notebooks/
│   ├── 00_profit_optimization.ipynb
│   ├── 01_survival_analysis.ipynb
│   ├── 02_walkforward_backtest.ipynb
│   ├── 03_shap_explainability.ipynb
│   ├── 04_fairness_audit.ipynb
│   └── 99_original_full_build.ipynb
├── src/
│   ├── config.py                  # paths + business constants
│   ├── data_loader.py             # load + clean + feature-engineer
│   ├── model.py                   # calibrated GBM PD model
│   ├── metrics.py                 # cost matrix, profit math, headline number
│   ├── decision.py                # production approve/reject engine
│   ├── explainer.py               # SHAP utilities
│   ├── survival.py                # KM curves + Cox PH
│   ├── backtest.py                # walk-forward temporal validation
│   ├── fairness.py                # disparate impact audit
│   └── train.py                   # end-to-end training entry point
├── outputs/                       # model.pkl, metrics.json, figures/
├── dashboard/
│   └── app.py                     # Streamlit app
├── requirements.txt
└── README.md
```

---

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Point at your LendingClub CSV (or edit src/config.py)
#    On Windows PowerShell:
$env:CREDIT_DATA_PATH = "C:\path\to\accepted_2007_to_2018Q4.csv"
#    On Linux/macOS:
export CREDIT_DATA_PATH=/path/to/accepted_2007_to_2018Q4.csv

# 3. Train — produces outputs/model.pkl, threshold.json, metrics.json
python -m src.train

# 4. Launch the dashboard
streamlit run dashboard/app.py
```

The dashboard's *Single applicant* tab takes a 6-field form and returns
approve / reject + expected $ P&L + SHAP explanation. The *Portfolio* tab
scores an uploaded batch CSV.

---

## Business assumptions (tunable in `src/config.py`)

| Parameter | Value | Meaning |
|---|---|---|
| `LGD` | 0.80 | Fraction of principal lost on default |
| `RECOVERY` | 0.20 | Fraction recovered via collections |
| `OPEX_RATE` | 0.01 | Origination + servicing cost (% of principal) |
| `COST_OF_CAP` | 0.04 | Risk-free opportunity cost (annualised) |
| `MIN_FICO` | 600 | Hard underwriting floor |
| `MAX_DTI` | 40.0 | Hard underwriting ceiling |
| `MAX_PD_OVERRIDE` | 0.55 | Never approve above this PD regardless of profit |

---

## The profit equation

For a loan of principal `L`, annual rate `r`, term `t` years:

```
E[Profit] = (1 − PD) · L · r · t   ←  expected interest income
          −  PD     · L · LGD       ←  expected loss on default
          −  L · OpEx              ←  servicing cost
```

Setting `E[Profit] = 0` and solving:

```
PD_breakeven = (r · t − OpEx) / (LGD + r · t)
```

A 20% / 60-month loan can absorb up to **55% PD** and still earn money;
a 7% / 36-month loan breaks even at just **20% PD**. That's why a single
fixed threshold leaves money on the table.

---

## Caveats interview-ers will probe

- **Grade is an input, not a protected class.** Disparate impact by grade
  is expected; we report it for transparency. Employment length is the
  more interesting fairness axis.
- **Survival times are approximated from `last_pymnt_d − issue_d`** — the
  dataset doesn't expose explicit charge-off dates.
- **LGD = 80% is industry-typical** but should be re-estimated from
  recoveries data in a live system. The break-even formula moves with it.
- **Walk-forward backtest uses an expanding training window.** Profit
  retention across 2015→2018 measures stability, not absolute generalization
  to a future macro regime.

---

## Tech stack

Python 3.10+ · scikit-learn · pandas · NumPy · lifelines · SHAP · Streamlit
· matplotlib · seaborn
