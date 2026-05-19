"""Streamlit dashboard for the Credit Decision Intelligence System.

Run:
    streamlit run dashboard/app.py

Requires that `python -m src.train` has been executed at least once
(populates outputs/model.pkl and outputs/feature_cols.json).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Make src importable when streamlit runs from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import config, data_loader, metrics as M, model as Mdl  # noqa: E402
from src import decision as D  # noqa: E402
from src import explainer as EXP  # noqa: E402


# ─── Streamlit setup ────────────────────────────────────────────────────────
st.set_page_config(page_title="Credit Decision Intelligence",
                   page_icon="💳", layout="wide")


@st.cache_resource(show_spinner=False)
def _load_model():
    if not config.MODEL_PATH.exists():
        return None
    return Mdl.load_model()


@st.cache_data(show_spinner=False)
def _load_metrics() -> dict:
    if not config.METRICS_PATH.exists():
        return {}
    return json.loads(config.METRICS_PATH.read_text())


def _empty_features_row(feature_cols: list[str]) -> dict:
    return {c: 0.0 for c in feature_cols}


def _build_application(form: dict, feature_cols: list[str]) -> pd.DataFrame:
    """Assemble a 1-row dataframe carrying both features and financials."""
    row = _empty_features_row(feature_cols)
    row.update({
        "loan_amnt": form["loan_amnt"],
        "int_rate": form["int_rate"],
        "term_months": form["term_months"],
        "annual_inc": form["annual_inc"],
        "dti": form["dti"],
        "fico": form["fico"],
        "emp_length_num": form["emp_length_num"],
        "grade_num": form["grade_num"],
        "subgrade_num": form["grade_num"] * 5 + 2,  # mid-grade default
        "purpose": form["purpose_code"],
        "home_ownership": 1,
        "application_type": 0,
        "verification_status": 1,
        "revol_util": 50.0,
        "credit_age_yrs": 10.0,
        "open_acc": 8,
        "total_acc": 20,
        "delinq_2yrs": 0,
        "inq_last_6mths": 1,
        "mort_acc": 1,
        "pub_rec": 0,
        "revol_bal": 8000.0,
        "tot_cur_bal": 50000.0,
        "tot_hi_cred_lim": 100000.0,
        "loan_to_income": form["loan_amnt"] / max(form["annual_inc"], 1.0),
        "payment_to_income": (form["loan_amnt"] / max(form["term_months"], 1))
                             / max(form["annual_inc"] / 12, 1.0),
        "debt_to_credit": 0.4,
        "acc_breadth": 0.4,
        "high_util_flag": 0,
        "state_freq": 0.02,
    })
    return pd.DataFrame([row])


# ─── Header ─────────────────────────────────────────────────────────────────
st.title("💳 Credit Decision Intelligence System")
st.caption("Profit-optimized loan approval — LendingClub 2007-2018")

trained = _load_model()
metrics = _load_metrics()

if trained is None:
    st.error(
        "**Model artifacts not found.** Run `python -m src.train` from the "
        "repo root first, then refresh this page."
    )
    st.stop()

# ─── Top KPI strip ──────────────────────────────────────────────────────────
policy = metrics.get("policy", {})
model_metrics = metrics.get("model", {})

c1, c2, c3, c4 = st.columns(4)
c1.metric("Extra profit / 1,000 loans",
          f"${policy.get('extra_profit_per_1000', 0):,.0f}",
          help="Profit-optimized policy vs blanket approval, on the holdout.")
c2.metric("Optimized approval rate",
          f"{policy.get('optimized_approval_rate', 0):.1%}")
c3.metric("Default rate among approved",
          f"{policy.get('optimized_default_rate', 0):.1%}")
c4.metric("Model AUC (test)",
          f"{model_metrics.get('auc_test', 0):.3f}")

st.divider()

# ─── Tabs ───────────────────────────────────────────────────────────────────
tab_single, tab_batch, tab_about = st.tabs(
    ["🧍 Single applicant", "📦 Portfolio (batch CSV)", "ℹ️ About the model"])


# ── Tab 1: single applicant ────────────────────────────────────────────────
with tab_single:
    st.subheader("Loan application")
    left, right = st.columns(2)
    with left:
        loan_amnt = st.number_input("Loan amount ($)", 1000, 40000, 15000, 500)
        int_rate = st.number_input("Interest rate (%)", 5.0, 30.0, 13.0, 0.1)
        term_months = st.selectbox("Term (months)", [36, 60], index=0)
        purpose = st.selectbox("Purpose", [
            "debt_consolidation", "credit_card", "home_improvement",
            "major_purchase", "medical", "small_business", "car", "other"])
    with right:
        annual_inc = st.number_input("Annual income ($)", 10000, 500000, 65000, 1000)
        dti = st.number_input("DTI (%)", 0.0, 50.0, 18.0, 0.5)
        fico = st.number_input("FICO score", 580, 850, 690, 1)
        emp_length = st.selectbox("Employment length", [
            "< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
            "6 years", "7 years", "8 years", "9 years", "10+ years"], index=5)
        grade = st.selectbox("LendingClub grade", list("ABCDEFG"), index=2)

    emp_map = data_loader.EMP_MAP
    grade_map = data_loader.GRADE_MAP
    purpose_map = {p: i for i, p in enumerate(sorted([
        "debt_consolidation", "credit_card", "home_improvement",
        "major_purchase", "medical", "small_business", "car", "other"]))}

    form = dict(loan_amnt=loan_amnt, int_rate=int_rate, term_months=term_months,
                annual_inc=annual_inc, dti=dti, fico=fico,
                emp_length_num=emp_map.get(emp_length, 1),
                grade_num=grade_map[grade], purpose_code=purpose_map[purpose])

    application = _build_application(form, trained.feature_cols)

    if st.button("Score application", type="primary"):
        dec = D.decide(application, trained)

        st.divider()
        c1, c2, c3 = st.columns(3)
        if dec.approve:
            c1.success(f"### ✅ APPROVE\n{dec.reason}")
        else:
            c1.error(f"### ❌ REJECT\n{dec.reason}")
        c2.metric("Predicted PD", f"{dec.pd_default:.1%}",
                  delta=f"break-even {dec.breakeven_pd:.1%}")
        c3.metric("Expected $ profit/loss",
                  f"${dec.expected_profit:,.0f}")

        c4, c5 = st.columns(2)
        c4.info(f"**Risk tier:** {dec.risk_tier}")
        c5.info(f"**Confidence:** {dec.confidence:.0%}")

        # ── SHAP explanation panel ─────────────────────────────────────
        st.subheader("Why this decision?")
        bundle = EXP.compute_shap(trained, application, max_samples=1)
        if bundle is None:
            st.warning("Install `shap` (`pip install shap`) to see the per-loan "
                       "contribution chart.")
            fb = EXP.fallback_importance(trained, trained.feature_cols).head(8)
            st.dataframe(fb, use_container_width=True)
        else:
            contribs = pd.DataFrame({
                "feature": bundle.feature_names,
                "value": bundle.X.iloc[0].values,
                "shap": bundle.values[0],
            }).assign(abs_shap=lambda d: d["shap"].abs())
            contribs = contribs.sort_values("abs_shap", ascending=False).head(8)
            contribs["direction"] = np.where(contribs["shap"] > 0,
                                             "↑ pushes toward default",
                                             "↓ pushes toward repay")
            st.dataframe(
                contribs[["feature", "value", "shap", "direction"]],
                use_container_width=True)


# ── Tab 2: batch ───────────────────────────────────────────────────────────
with tab_batch:
    st.subheader("Portfolio simulator")
    st.markdown(
        "Upload a CSV with columns matching the feature contract "
        "(`outputs/feature_cols.json`) plus `loan_amnt`, `int_rate`, "
        "`term_months`. The simulator scores every row and reports "
        "portfolio-level profit under the profit-optimized policy.")

    sample_path = config.CLEAN_PARQUET_PATH
    csv_fallback = sample_path.with_suffix(".csv")
    sample_df = None
    if sample_path.exists():
        try:
            sample_df = pd.read_parquet(sample_path)
        except Exception:
            pass
    if sample_df is None and csv_fallback.exists():
        sample_df = pd.read_csv(csv_fallback)

    use_sample = st.checkbox("Use built-in sample portfolio "
                             "(from `outputs/clean.parquet`)",
                             value=sample_df is not None)
    uploaded = st.file_uploader("Or upload your own CSV", type=["csv"])

    portfolio = None
    if uploaded is not None:
        portfolio = pd.read_csv(uploaded)
    elif use_sample and sample_df is not None:
        portfolio = sample_df

    if portfolio is not None and len(portfolio):
        missing = [c for c in trained.feature_cols if c not in portfolio.columns]
        if missing:
            st.error(f"CSV is missing {len(missing)} required feature columns. "
                     f"First 10 missing: {missing[:10]}")
        else:
            portfolio = portfolio.copy()
            portfolio["pd"] = trained.predict_pd(portfolio)
            portfolio["exp_profit"] = M.expected_profit(
                portfolio["pd"], portfolio["loan_amnt"],
                portfolio["int_rate"], portfolio["term_months"])
            portfolio["approve"] = portfolio["exp_profit"] > 0

            approved = portfolio[portfolio["approve"]]
            total_exp = float(approved["exp_profit"].sum())
            naive_exp = float(portfolio["exp_profit"].sum())

            c1, c2, c3 = st.columns(3)
            c1.metric("Loans in portfolio", f"{len(portfolio):,}")
            c2.metric("Approved", f"{len(approved):,}",
                      delta=f"{len(approved)/len(portfolio):.0%}")
            c3.metric("Expected portfolio profit",
                      f"${total_exp:,.0f}",
                      delta=f"${total_exp - naive_exp:,.0f} vs approve-all")

            st.bar_chart(portfolio.groupby(
                pd.cut(portfolio["pd"], bins=10))["approve"].mean()
                .reset_index(drop=True))

            st.dataframe(
                portfolio[["loan_amnt", "int_rate", "term_months",
                           "pd", "exp_profit", "approve"]].head(50),
                use_container_width=True)

            st.download_button(
                "Download scored portfolio (CSV)",
                portfolio.to_csv(index=False).encode("utf-8"),
                file_name="scored_portfolio.csv", mime="text/csv")


# ── Tab 3: about ───────────────────────────────────────────────────────────
with tab_about:
    st.markdown(f"""
### What this app does
Given a loan application, the system predicts probability of default (PD)
with a calibrated gradient-boosted model, derives the **loan-specific
break-even PD** from the cost matrix, and approves only if expected dollar
profit is positive.

### Why this is different from a typical credit model
Most models optimize **AUC**. This one optimizes **dollars**. The cost
matrix is asymmetric — rejecting a profitable borrower costs us net
interest revenue; approving a defaulter costs us LGD × principal.
That asymmetry, not 0.5, is what sets the threshold.

### Economics
- **LGD** = {config.LGD:.0%} (loss given default)
- **OpEx** = {config.OPEX_RATE:.0%} of principal
- **Cost of capital** = {config.COST_OF_CAP:.0%} p.a.
- **Hard floors:** FICO ≥ {config.MIN_FICO}, DTI ≤ {config.MAX_DTI},
  PD ≤ {config.MAX_PD_OVERRIDE:.0%}

### Model
- Calibrated Gradient Boosting (isotonic) → outputs are valid probabilities
- Trained on {model_metrics.get('n_train', '—')} loans;
  AUC test {model_metrics.get('auc_test', 0):.3f}

### Reproducing
```bash
pip install -r requirements.txt
python -m src.train
streamlit run dashboard/app.py
```
""")
