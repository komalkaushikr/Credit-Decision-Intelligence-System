"""Production decision engine: turn a single loan application into an
approve/reject + expected dollar P&L + risk tier.

This is the function the Streamlit app calls. Mirrors Phase 8 of the
original notebook but uses the modular pieces in src/.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd

from . import config
from . import metrics as M
from .model import TrainedModel


@dataclass
class LoanDecision:
    approve: bool
    reason: str
    pd_default: float
    breakeven_pd: float
    expected_profit: float
    risk_tier: str
    confidence: float  # |breakeven - pd| / breakeven, clipped to [0,1]


def _risk_tier(pd_default: float) -> str:
    if pd_default < 0.05: return "Prime"
    if pd_default < 0.15: return "Near-prime"
    if pd_default < 0.30: return "Subprime"
    if pd_default < 0.50: return "Deep subprime"
    return "Reject zone"


def decide(application: pd.DataFrame, model: TrainedModel,
           min_fico: float = config.MIN_FICO,
           max_dti: float = config.MAX_DTI,
           max_pd_override: float = config.MAX_PD_OVERRIDE) -> LoanDecision:
    """`application` is a single-row dataframe carrying both the model
    features and the financial columns (loan_amnt, int_rate, term_months)."""
    pd_default = float(model.predict_pd(application)[0])

    L = float(application["loan_amnt"].iloc[0])
    r = float(application["int_rate"].iloc[0])
    t = float(application["term_months"].iloc[0])
    exp_profit = float(M.expected_profit(pd_default, L, r, t))
    be = M.breakeven_pd(r, t)

    # Hard floors.
    fico = float(application["fico"].iloc[0]) if "fico" in application.columns else None
    dti = float(application["dti"].iloc[0]) if "dti" in application.columns else None
    if fico is not None and fico < min_fico:
        return LoanDecision(False, f"FICO {fico:.0f} below floor {min_fico}",
                            pd_default, be, exp_profit, _risk_tier(pd_default), 1.0)
    if dti is not None and dti > max_dti:
        return LoanDecision(False, f"DTI {dti:.1f} above ceiling {max_dti}",
                            pd_default, be, exp_profit, _risk_tier(pd_default), 1.0)
    if pd_default > max_pd_override:
        return LoanDecision(False, f"PD {pd_default:.1%} above hard ceiling "
                            f"{max_pd_override:.0%}",
                            pd_default, be, exp_profit, _risk_tier(pd_default), 1.0)

    approve = exp_profit > 0
    reason = ("Expected profit positive — approve."
              if approve else
              f"Expected profit ${exp_profit:,.0f} — below zero, reject.")
    confidence = min(1.0, abs(be - pd_default) / max(be, 1e-6))
    return LoanDecision(approve=approve, reason=reason,
                        pd_default=pd_default, breakeven_pd=be,
                        expected_profit=exp_profit,
                        risk_tier=_risk_tier(pd_default),
                        confidence=float(confidence))


def to_dict(decision: LoanDecision) -> dict[str, Any]:
    return asdict(decision)
