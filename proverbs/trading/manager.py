"""Trading manager — selects the broker, enforces the LIVE gate, executes orders.

The single choke point between signals and money:

    decision (BUY/HOLD/REDUCE)
        -> size the order
        -> RiskManager.check(...)            (kill switch, limits, market hours)
        -> LIVE gate: if not state.live -> DRY-RUN (log intended order, send nothing)
        -> broker.submit_order(...)          (paper or Robinhood)

Safe by default: BROKER=paper, LIVE_TRADING=false.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

from proverbs.brains.orchestrator import BUY as ACTION_BUY, REDUCE as ACTION_REDUCE, Decision
from proverbs.config import settings
from proverbs.trading.base import BUY, SELL, DRY_RUN, REJECTED, Broker, OrderResult
from proverbs.trading.paper_broker import PaperBroker
from proverbs.trading.risk import RiskManager, TradingState

logger = logging.getLogger(__name__)


def _make_broker() -> Broker:
    if settings.broker == "robinhood":
        from proverbs.trading.robinhood_broker import RobinhoodBroker

        return RobinhoodBroker()
    return PaperBroker()


class TradingManager:
    def __init__(self) -> None:
        self.state = TradingState()
        self.risk = RiskManager(self.state)
        self.broker: Optional[Broker] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------- lifecycle
    def init(self) -> None:
        """Create and connect the broker. Never raises — logs and degrades."""
        self.broker = _make_broker()
        try:
            ok = self.broker.connect()
        except Exception:
            logger.exception("TradingManager: broker connect crashed.")
            ok = False
        if not ok:
            logger.warning("TradingManager: broker '%s' not connected; trading disabled.",
                           self.broker.name)
            self.state.trading_enabled = False
            return
        acc = self.broker.get_account()
        if acc:
            self.state.start_equity = acc.equity
        logger.info("TradingManager ready: broker=%s mode=%s trading_enabled=%s",
                    self.broker.name, self.state.summary()["mode"], self.state.trading_enabled)

    # --------------------------------------------------------------- sizing
    def _buy_notional(self, decision: Decision) -> float:
        # Scale base size by confidence, capped by the per-order limit.
        size = settings.base_order_notional * (0.5 + 0.5 * decision.confidence)
        return round(min(size, settings.max_order_notional), 2)

    def _sell_notional(self, symbol: str) -> float:
        pos = self.broker.get_position(symbol) if self.broker else None
        if not pos:
            return 0.0
        # Reduce by the base order size, but never more than we hold.
        return round(min(settings.base_order_notional, pos.market_value), 2)

    # -------------------------------------------------------------- execute
    def execute_decision(self, decision: Decision) -> Optional[OrderResult]:
        """Turn a BUY/REDUCE decision into an order (or a dry-run/rejection).

        Returns None for HOLD (nothing to do).
        """
        if self.broker is None:
            return None
        if decision.action == ACTION_BUY:
            side, notional = BUY, self._buy_notional(decision)
        elif decision.action == ACTION_REDUCE:
            side, notional = SELL, self._sell_notional(decision.symbol)
            if notional <= 0:
                return None  # nothing held to reduce
        else:
            return None
        return self._place(decision.symbol, side, notional, decision.confidence)

    def manual_order(self, symbol: str, side: str, notional: float) -> OrderResult:
        """Explicit order from a command; confidence treated as passing."""
        side = side.lower()
        confidence = max(settings.order_confidence_min, 1.0)
        return self._place(symbol, side, round(notional, 2), confidence)

    def _place(self, symbol: str, side: str, notional: float, confidence: float) -> OrderResult:
        with self._lock:
            allowed, reason = self.risk.check(self.broker, symbol, side, notional, confidence)
            if not allowed:
                logger.info("Order rejected: %s %s $%.2f — %s", side, symbol, notional, reason)
                return OrderResult(status=REJECTED, symbol=symbol.upper(), side=side,
                                   notional=notional, message=reason)

            price = self.broker.get_price(symbol) or 0.0

            if not self.state.live:
                logger.info("[DRY-RUN] would %s $%.2f of %s @ $%.2f", side, notional, symbol, price)
                return OrderResult(status=DRY_RUN, symbol=symbol.upper(), side=side,
                                   notional=notional, price=price,
                                   quantity=(notional / price if price else 0.0),
                                   message="Dry-run: order logged, not sent. Set LIVE_TRADING=true to trade.")

            result = self.broker.submit_order(symbol, side, notional, price)
            logger.info("Order %s: %s %s $%.2f -> %s", result.status, side, symbol, notional, result.message)
            return result

    # ---------------------------------------------------------------- state
    def halt(self) -> None:
        self.state.trading_enabled = False

    def resume(self) -> None:
        self.state.trading_enabled = True

    def set_live(self, live: bool) -> None:
        self.state.live = live

    def status(self) -> dict:
        broker_name = self.broker.name if self.broker else settings.broker
        acc = self.broker.get_account().to_dict() if (self.broker and self.state.trading_enabled) else None
        return {
            "broker": broker_name,
            **self.state.summary(),
            "account": acc,
            "limits": {
                "base_order_notional": settings.base_order_notional,
                "max_order_notional": settings.max_order_notional,
                "max_position_notional": settings.max_position_notional,
                "max_open_positions": settings.max_open_positions,
                "daily_loss_limit": settings.daily_loss_limit,
                "order_confidence_min": settings.order_confidence_min,
            },
        }


# Single shared manager for the process.
trading = TradingManager()
