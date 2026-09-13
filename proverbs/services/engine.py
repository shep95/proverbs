"""The engine — one full analysis cycle, end to end.

Wires the three brains + feeds + persistence together (build step 9 of the plan):

    for each symbol in the watchlist:
        fiscal  = FiscalAI.analyze(symbol)      # quant signal + risk metrics
        cultural= CulturalAI.analyze(symbol)    # sentiment index
        decision= Orchestrator.decide(...)      # BUY / HOLD / REDUCE
        persist the signal (audit trail)

    then, on the shared account:
        apply simulated balance growth (risk tier x average signal)
        run the auto-withdrawal check and persist any withdrawal

This module also exposes the account operations (deposit / withdraw / set risk /
portfolio) shared by both the Discord bot and the Flask API, so the two
interfaces stay perfectly consistent.

NOTE: this is a *paper-trading simulation*. It never places real brokerage
orders; balances and growth are simulated for education/experimentation.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from proverbs.brains.cultural_ai import CulturalAI
from proverbs.brains.fiscal_ai import FiscalAI
from proverbs.brains.orchestrator import Decision, Orchestrator
from proverbs.config import settings
from proverbs.feeds.market_feed import latest_snapshot
from proverbs.persistence import db
from proverbs.persistence.models import Account
from proverbs.trading.base import OrderResult
from proverbs.trading.manager import trading
from proverbs.trading.session import sessions

logger = logging.getLogger(__name__)

RISK_FACTORS = {"Low": 0.01, "Medium": 0.02, "High": 0.04}
VALID_RISK = tuple(RISK_FACTORS.keys())


@dataclass
class WithdrawalEvent:
    account: str
    amount: float
    balance_after: float


@dataclass
class CycleReport:
    decisions: List[Decision] = field(default_factory=list)
    avg_score: float = 0.0
    simulated_return_pct: float = 0.0
    balance_before: float = 0.0
    balance_after: float = 0.0
    withdrawals: List[WithdrawalEvent] = field(default_factory=list)
    orders: List[OrderResult] = field(default_factory=list)
    triggered_alerts: List[dict] = field(default_factory=list)
    graded: int = 0
    errors: List[str] = field(default_factory=list)


class Engine:
    def __init__(self) -> None:
        self.fiscal = FiscalAI()
        self.cultural = CulturalAI()
        self.orchestrator = Orchestrator()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ cycle
    def run_cycle(self, symbols: Optional[List[str]] = None) -> CycleReport:
        """Run a full analysis cycle. Thread-safe (scheduler + manual triggers)."""
        if symbols is None:
            with db.session_scope() as s:
                symbols = db.get_watchlist(s)
        with self._lock:
            report = CycleReport()
            scores: List[float] = []
            prices: Dict[str, float] = {}       # symbol -> latest price (for grading)
            score_map: Dict[str, float] = {}    # symbol -> combined score (for alerts)

            for symbol in symbols:
                try:
                    fiscal = self.fiscal.analyze(symbol)
                    cultural = self.cultural.analyze(symbol)
                    decision = self.orchestrator.decide(fiscal, cultural, symbol)
                    report.decisions.append(decision)
                    scores.append(decision.combined_score)
                    score_map[symbol.upper()] = decision.combined_score
                    if fiscal and fiscal.last_price > 0:
                        prices[symbol.upper()] = fiscal.last_price
                    with db.session_scope() as s:
                        db.save_signal(s, decision)
                    # Route the decision to the broker (dry-run/paper by default).
                    # Always trade during an active paper session, even if AUTO_TRADE is off.
                    if settings.auto_trade or sessions.active():
                        try:
                            order = trading.execute_decision(decision)
                            if order is not None:
                                report.orders.append(order)
                        except Exception:
                            logger.exception("Engine: order execution failed for %s", symbol)
                except Exception as exc:  # never let one bad ticker kill the cycle
                    logger.exception("Engine: cycle failed for %s", symbol)
                    report.errors.append(f"{symbol}: {exc}")

            report.avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0
            # Grade older signals against the fresh prices, and fire any alerts.
            report.graded = self._grade_signals(prices)
            report.triggered_alerts = self._evaluate_alerts(score_map)
            self._apply_simulation(report)
            # Snapshot the paper session's equity curve and auto-expire if due.
            try:
                sessions.on_cycle()
            except Exception:
                logger.exception("Engine: session snapshot failed")
            logger.info("Engine cycle complete: %d signals, avg=%.3f, return=%.3f%%",
                        len(report.decisions), report.avg_score, report.simulated_return_pct)
            return report

    def _apply_simulation(self, report: CycleReport) -> None:
        """Grow/shrink the shared account balance based on risk tier x avg signal."""
        with db.session_scope() as s:
            acc = db.get_or_create_account(s, db.GLOBAL_ACCOUNT_ID, "Global")
            report.balance_before = round(acc.balance, 2)
            if acc.balance > 0:
                risk_factor = RISK_FACTORS.get(acc.risk_level, 0.02)
                noise = random.uniform(-0.25, 0.25) * risk_factor
                period_return = report.avg_score * risk_factor + noise
                acc.balance = max(0.0, acc.balance * (1 + period_return))
                report.simulated_return_pct = round(period_return * 100, 4)

            # Auto-withdrawal on profit milestone.
            initial = acc.total_deposited - acc.total_withdrawn
            result = self.orchestrator.check_auto_withdrawal(initial, acc.balance)
            if result.triggered:
                acc.balance = result.new_balance
                acc.total_withdrawn += result.amount
                db.record_transaction(s, acc, "auto_withdrawal", result.amount,
                                      note=f"Auto-withdrawal at {settings.auto_withdraw_multiple}x deposit")
                report.withdrawals.append(
                    WithdrawalEvent(account=acc.display_name, amount=result.amount,
                                    balance_after=acc.balance))
                logger.info("Auto-withdrawal triggered: $%.2f", result.amount)

            report.balance_after = round(acc.balance, 2)

    # ------------------------------------------------------- accuracy grading
    def _grade_signals(self, prices: Dict[str, float]) -> int:
        """Grade ungraded signals older than the horizon against the fresh price."""
        if not prices:
            return 0
        now = time.time()
        graded = 0
        with db.session_scope() as s:
            for symbol, price in prices.items():
                if price <= 0:
                    continue
                # Per-symbol horizon override, else the global default.
                horizon = settings.horizon_hours_for(symbol) * 3600
                cutoff_dt = datetime.utcfromtimestamp(now - horizon)
                for sig in db.ungraded_signals(s, symbol, cutoff_dt):
                    actual_up = price > sig.last_price
                    predicted_up = sig.predicted_up == 1
                    sig.correct = 1 if actual_up == predicted_up else 0
                    sig.outcome_price = price
                    sig.outcome_return = (price - sig.last_price) / sig.last_price if sig.last_price else 0.0
                    sig.graded_ts = now
                    graded += 1
        if graded:
            logger.info("Engine: graded %d past signal(s).", graded)
        return graded

    def _evaluate_alerts(self, score_map: Dict[str, float]) -> List[dict]:
        """Fire any per-user score alerts crossed this cycle (one-shot)."""
        if not score_map:
            return []
        triggered: List[dict] = []
        now = time.time()
        with db.session_scope() as s:
            for a in db.active_alerts(s):
                if a.symbol not in score_map:
                    continue
                score = score_map[a.symbol]
                hit = ((a.direction == "above" and score >= a.threshold) or
                       (a.direction == "below" and score <= a.threshold))
                if hit:
                    a.active = False
                    a.fired_ts = now
                    triggered.append({"user_id": a.user_id, "channel_id": a.channel_id,
                                      "symbol": a.symbol, "direction": a.direction,
                                      "threshold": round(a.threshold, 4), "score": round(score, 4)})
        return triggered

    def accuracy_stats(self, symbol: Optional[str] = None) -> dict:
        with db.session_scope() as s:
            return db.accuracy_stats(s, symbol)

    # ------------------------------------------------------- watchlist/weights
    def watchlist(self) -> List[str]:
        with db.session_scope() as s:
            return db.get_watchlist(s)

    def add_to_watchlist(self, symbol: str, added_by: str = "") -> bool:
        with db.session_scope() as s:
            return db.add_watch(s, symbol, added_by)

    def remove_from_watchlist(self, symbol: str) -> bool:
        with db.session_scope() as s:
            return db.remove_watch(s, symbol)

    def set_weights(self, fiscal: float, cultural: float) -> Dict:
        if fiscal < 0 or cultural < 0 or (fiscal + cultural) <= 0:
            return {"status": "error", "message": "Weights must be non-negative and not both zero."}
        settings.fiscal_weight = float(fiscal)
        settings.cultural_weight = float(cultural)
        fw, cw = settings.normalised_weights
        return {"status": "success", "fiscal_weight": round(fw, 4), "cultural_weight": round(cw, 4),
                "note": "Runtime override (resets to env defaults on restart)."}

    def add_alert(self, user_id: str, channel_id: str, symbol: str,
                  direction: str, threshold: float) -> Dict:
        if direction.lower() not in ("above", "below"):
            return {"status": "error", "message": "Direction must be 'above' or 'below'."}
        with db.session_scope() as s:
            alert = db.add_alert(s, user_id, channel_id, symbol, direction, threshold)
            return {"status": "success", **alert.to_dict()}

    def list_alerts(self, user_id: str) -> List[dict]:
        with db.session_scope() as s:
            return [a.to_dict() for a in db.list_alerts(s, user_id)]

    # --------------------------------------------------------------- accounts
    def deposit(self, discord_user_id: str, amount: float, display_name: str = "") -> Dict:
        if amount <= 0:
            return {"status": "error", "message": "Deposit amount must be positive."}
        with db.session_scope() as s:
            acc = db.get_or_create_account(s, discord_user_id, display_name)
            acc.balance += amount
            acc.total_deposited += amount
            db.record_transaction(s, acc, "deposit", amount)
            return {"status": "success", **acc.to_dict()}

    def withdraw(self, discord_user_id: str, amount: float, display_name: str = "") -> Dict:
        if amount <= 0:
            return {"status": "error", "message": "Withdrawal amount must be positive."}
        with db.session_scope() as s:
            acc = db.get_or_create_account(s, discord_user_id, display_name)
            if amount > acc.balance:
                return {"status": "error", "message": "Insufficient balance."}
            acc.balance -= amount
            acc.total_withdrawn += amount
            db.record_transaction(s, acc, "withdrawal", amount)
            return {"status": "success", **acc.to_dict()}

    def set_risk(self, discord_user_id: str, risk_level: str, display_name: str = "") -> Dict:
        risk_level = risk_level.capitalize()
        if risk_level not in VALID_RISK:
            return {"status": "error", "message": f"Risk must be one of {', '.join(VALID_RISK)}."}
        with db.session_scope() as s:
            acc = db.get_or_create_account(s, discord_user_id, display_name)
            acc.risk_level = risk_level
            return {"status": "success", **acc.to_dict()}

    def get_account(self, discord_user_id: str, display_name: str = "") -> Dict:
        with db.session_scope() as s:
            acc = db.get_or_create_account(s, discord_user_id, display_name)
            return acc.to_dict()

    def get_portfolio(self, discord_user_id: str) -> Dict:
        with db.session_scope() as s:
            acc = db.get_or_create_account(s, discord_user_id)
            positions = [p.to_dict() for p in acc.positions]
            return {**acc.to_dict(), "positions": positions}

    def get_history(self, discord_user_id: str, limit: int = 15) -> List[Dict]:
        with db.session_scope() as s:
            acc = db.get_or_create_account(s, discord_user_id)
            return [t.to_dict() for t in db.recent_transactions(s, acc, limit)]

    def latest_signals(self) -> List[Dict]:
        with db.session_scope() as s:
            return [sig.to_dict() for sig in db.latest_signal_per_symbol(s)]

    def signal_history(self, symbol: Optional[str] = None, limit: int = 20) -> List[Dict]:
        with db.session_scope() as s:
            return [sig.to_dict() for sig in db.recent_signals(s, symbol, limit)]

    def price_snapshot(self, symbol: str) -> Optional[Dict]:
        snap = latest_snapshot(symbol)
        if not snap:
            return None
        return {"symbol": snap.symbol, "last_price": round(snap.last_price, 4),
                "change_pct": round(snap.change_pct, 2)}


# A single shared engine instance for the whole process.
engine = Engine()
