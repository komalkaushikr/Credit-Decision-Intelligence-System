"""Profit-based metrics: cost matrix, expected profit, optimal threshold.

The headline number recruiters care about lives here:
`extra_profit_vs_naive_per_1000(...)` — how many extra dollars the
profit-optimized policy earns per 1,000 loans vs blanket approval.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


# ─── Per-loan economics ─────────────────────────────────────────────────────

def expected_profit(pd_default: np.ndarray | float,
                    loan_amnt: np.ndarray | float,
                    int_rate_pct: np.ndarray | float,
                    term_months: np.ndarray | float,
                    lgd: float = config.LGD,
                    opex: float = config.OPEX_RATE) -> np.ndarray | float:
    """Expected dollar profit per loan under the standard model.

    E[Profit] = (1 - PD) * L * r * t  -  PD * L * LGD  -  L * OpEx
    """
    r = np.asarray(int_rate_pct) / 100.0
    t = np.asarray(term_months) / 12.0
    L = np.asarray(loan_amnt)
    p = np.asarray(pd_default)
    return (1 - p) * L * r * t - p * L * lgd - L * opex


def realised_profit(default: np.ndarray | int,
                    loan_amnt: np.ndarray | float,
                    int_rate_pct: np.ndarray | float,
                    term_months: np.ndarray | float,
                    lgd: float = config.LGD,
                    opex: float = config.OPEX_RATE) -> np.ndarray:
    """Realised dollar P&L on an approved loan given the *actual* outcome."""
    r = np.asarray(int_rate_pct) / 100.0
    t = np.asarray(term_months) / 12.0
    L = np.asarray(loan_amnt)
    d = np.asarray(default)
    interest = L * r * t
    loss = L * lgd
    return np.where(d == 1, -loss, interest) - L * opex


def breakeven_pd(int_rate_pct: float, term_months: float,
                 lgd: float = config.LGD, opex: float = config.OPEX_RATE) -> float:
    """PD above which the loan loses money in expectation."""
    r = int_rate_pct / 100.0
    t = term_months / 12.0
    return float(max(0.0, min(1.0, (r * t - opex) / (lgd + r * t))))


# ─── Cost matrix (FP / FN framing) ──────────────────────────────────────────

@dataclass
class CostMatrix:
    """Per-loan dollar costs of each confusion-matrix cell.

    Positive class = "will default".

    - FP (predict default, actually good) → opportunity cost of rejecting a
      profitable borrower = interest revenue we would have earned net of OpEx.
    - FN (predict good, actually default) → loss given default + OpEx.
    - TP (correctly reject) → 0 (we did not lend).
    - TN (correctly approve) → negative cost = realised profit.
    """
    tn_value: float   # value of approving a good loan (positive)
    fp_cost: float    # cost of rejecting a good loan (positive)
    fn_cost: float    # cost of approving a defaulter (positive)
    tp_value: float = 0.0


def per_loan_cost_matrix(loan_amnt: float, int_rate_pct: float,
                         term_months: float,
                         lgd: float = config.LGD,
                         opex: float = config.OPEX_RATE) -> CostMatrix:
    interest = loan_amnt * (int_rate_pct / 100) * (term_months / 12)
    opex_dollars = loan_amnt * opex
    tn_value = interest - opex_dollars
    fp_cost = interest - opex_dollars            # forgone net interest
    fn_cost = loan_amnt * lgd + opex_dollars     # principal loss + OpEx
    return CostMatrix(tn_value=tn_value, fp_cost=fp_cost,
                      fn_cost=fn_cost, tp_value=0.0)


def optimal_threshold_analytical(int_rate_pct: float, term_months: float,
                                 lgd: float = config.LGD,
                                 opex: float = config.OPEX_RATE) -> float:
    """Closed-form optimal PD threshold from the cost matrix.

    Approve iff E[profit] > 0  ⇔  PD < (FP_cost) / (FP_cost + FN_cost)
    which reduces to the break-even PD formula.
    """
    cm = per_loan_cost_matrix(1.0, int_rate_pct, term_months, lgd, opex)
    return cm.fp_cost / (cm.fp_cost + cm.fn_cost)


# ─── Threshold search & portfolio comparison ────────────────────────────────

def sweep_threshold(df: pd.DataFrame, pd_col: str = "pd",
                    n_steps: int = 101) -> pd.DataFrame:
    """For each threshold t in [0,1], compute realised portfolio profit if
    we approved loans where PD < t. Used to plot the profit curve."""
    out = []
    for t in np.linspace(0, 1, n_steps):
        approved = df[df[pd_col] < t]
        if len(approved) == 0:
            out.append((t, 0.0, 0, 0.0))
            continue
        profit = realised_profit(approved["default"].values,
                                 approved["loan_amnt"].values,
                                 approved["int_rate"].values,
                                 approved["term_months"].values).sum()
        out.append((t, profit, len(approved),
                    approved["default"].mean()))
    return pd.DataFrame(out, columns=["threshold", "profit",
                                      "n_approved", "default_rate"])


def find_optimal_fixed_threshold(df: pd.DataFrame, pd_col: str = "pd") -> dict:
    sweep = sweep_threshold(df, pd_col)
    best = sweep.loc[sweep["profit"].idxmax()]
    return {"threshold": float(best["threshold"]),
            "profit": float(best["profit"]),
            "n_approved": int(best["n_approved"]),
            "default_rate": float(best["default_rate"])}


# ─── Headline metric: extra profit vs naive ─────────────────────────────────

def policy_profit(df: pd.DataFrame, approve_mask: np.ndarray) -> float:
    if approve_mask.sum() == 0:
        return 0.0
    sub = df.loc[approve_mask]
    return float(realised_profit(sub["default"].values,
                                 sub["loan_amnt"].values,
                                 sub["int_rate"].values,
                                 sub["term_months"].values).sum())


def per_loan_profit(total_profit: float, n_loans: int, scale: int = 1000) -> float:
    if n_loans == 0:
        return 0.0
    return total_profit / n_loans * scale


def compare_to_naive(df: pd.DataFrame, pd_col: str = "pd") -> dict:
    """Compute the headline number.

    Naive = approve every applicant. Optimized = approve iff per-loan
    expected profit (using predicted PD) is positive — i.e. PD below the
    loan-specific break-even.
    """
    n = len(df)
    naive_mask = np.ones(n, dtype=bool)
    naive_profit = policy_profit(df, naive_mask)

    exp_profit = expected_profit(df[pd_col].values,
                                 df["loan_amnt"].values,
                                 df["int_rate"].values,
                                 df["term_months"].values)
    opt_mask = exp_profit > 0
    opt_profit = policy_profit(df, opt_mask)

    return {
        "n_loans": int(n),
        "naive_profit": naive_profit,
        "naive_approval_rate": 1.0,
        "naive_default_rate": float(df["default"].mean()),
        "optimized_profit": opt_profit,
        "optimized_approval_rate": float(opt_mask.mean()),
        "optimized_default_rate": float(df.loc[opt_mask, "default"].mean())
            if opt_mask.sum() else 0.0,
        "extra_profit_total": opt_profit - naive_profit,
        "extra_profit_per_1000": per_loan_profit(opt_profit - naive_profit, n, 1000),
        "optimized_profit_per_1000": per_loan_profit(opt_profit, n, 1000),
        "naive_profit_per_1000": per_loan_profit(naive_profit, n, 1000),
    }


def headline_summary(stats: dict) -> str:
    """Recruiter-friendly one-liner."""
    return (
        f"Profit-optimized policy generates ${stats['extra_profit_per_1000']:,.0f} "
        f"more profit per 1,000 loans than blanket approval "
        f"(${stats['optimized_profit_per_1000']:,.0f} vs "
        f"${stats['naive_profit_per_1000']:,.0f}). "
        f"Approval rate {stats['optimized_approval_rate']:.0%}, "
        f"default rate among approved {stats['optimized_default_rate']:.1%}."
    )
