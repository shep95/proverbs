"""Orchestrator — the decision brain.

Phase 4 of the rebuild: combine the fiscal and cultural signals with tunable
weights into a single combined score, map it to a BUY / HOLD / REDUCE action,
and own the auto-withdrawal rule. Pure logic — persistence lives elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from proverbs.brains.cultural_ai import CulturalInsight
from proverbs.brains.fiscal_ai import FiscalInsight
from proverbs.config import settings

BUY = "BUY"
HOLD = "HOLD"
REDUCE = "REDUCE"


@dataclass
class Decision:
    symbol: str
    action: str                 # BUY | HOLD | REDUCE
    combined_score: float       # weighted score in [-1, 1]
    confidence: float           # blended confidence in [0, 1]
    fiscal_signal: float
    cultural_signal: float
    fiscal: Optional[FiscalInsight] = None
    cultural: Optional[CulturalInsight] = None

    @property
    def emoji(self) -> str:
        return {"BUY": "🟢", "HOLD": "🟡", "REDUCE": "🔴"}.get(self.action, "⚪")

    def rationale(self) -> str:
        return (
            f"fiscal={self.fiscal_signal:+.2f} (w={settings.normalised_weights[0]:.2f}), "
            f"cultural={self.cultural_signal:+.2f} (w={settings.normalised_weights[1]:.2f}) "
            f"-> combined={self.combined_score:+.2f}"
        )


@dataclass
class WithdrawalResult:
    triggered: bool
    amount: float
    new_balance: float


class Orchestrator:
    name = "Decision Orchestrator"

    def decide(self, fiscal: Optional[FiscalInsight], cultural: Optional[CulturalInsight],
               symbol: str) -> Decision:
        fw, cw = settings.normalised_weights
        f_sig = fiscal.signal if fiscal else 0.0
        c_sig = cultural.index if cultural else 0.0

        combined = fw * f_sig + cw * c_sig
        combined = max(-1.0, min(1.0, combined))

        if combined >= settings.buy_threshold:
            action = BUY
        elif combined <= settings.reduce_threshold:
            action = REDUCE
        else:
            action = HOLD

        f_conf = fiscal.confidence if fiscal else 0.0
        c_conf = min(1.0, (cultural.sample_size / 10.0)) if cultural else 0.0
        confidence = fw * f_conf + cw * c_conf

        return Decision(
            symbol=symbol.upper(),
            action=action,
            combined_score=round(combined, 4),
            confidence=round(confidence, 4),
            fiscal_signal=round(f_sig, 4),
            cultural_signal=round(c_sig, 4),
            fiscal=fiscal,
            cultural=cultural,
        )

    def check_auto_withdrawal(self, initial_deposit: float, balance: float) -> WithdrawalResult:
        """Auto-withdraw when balance >= deposit * MULTIPLE; pull FRACTION of profit."""
        multiple = settings.auto_withdraw_multiple
        fraction = settings.auto_withdraw_fraction
        if initial_deposit > 0 and balance >= initial_deposit * multiple:
            profit = balance - initial_deposit
            amount = profit * fraction
            return WithdrawalResult(triggered=True, amount=round(amount, 2),
                                    new_balance=round(balance - amount, 2))
        return WithdrawalResult(triggered=False, amount=0.0, new_balance=round(balance, 2))
