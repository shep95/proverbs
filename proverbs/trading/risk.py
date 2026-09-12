"""Risk manager, market-hours guard, and runtime trading state.

Every order — auto or manual — passes through ``RiskManager.check`` before it can
reach a broker. The runtime state (kill switch + live/dry-run) is mutable at
runtime via Discord commands, so you can halt trading without a redeploy.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Optional, Tuple

from proverbs.config import settings
from proverbs.trading.base import BUY, Broker

logger = logging.getLogger(__name__)


def is_market_open(now: Optional[datetime] = None) -> bool:
    """True during regular US equity hours (Mon-Fri, 09:30-16:00 ET).

    NOTE: exchange holidays are NOT tracked — orders on a holiday will be
    rejected by the broker. Set IGNORE_MARKET_HOURS=true to bypass this guard.
    """
    if settings.ignore_market_hours:
        return True
    try:
        from zoneinfo import ZoneInfo

        et = ZoneInfo("America/New_York")
    except Exception:
        return True  # can't determine tz -> don't block
    now = now or datetime.now(et)
    if now.tzinfo is None:
        now = now.replace(tzinfo=et)
    now = now.astimezone(et)
    if now.weekday() >= 5:  # Sat/Sun
        return False
    return time(9, 30) <= now.time() <= time(16, 0)


@dataclass
class TradingState:
    """Runtime, mutable flags. Initialised from config; changed by /kill, /mode."""
    trading_enabled: bool = field(default_factory=lambda: settings.trading_enabled)
    live: bool = field(default_factory=lambda: settings.live_trading)
    start_equity: Optional[float] = None  # recorded at connect for the loss guard

    def summary(self) -> dict:
        return {"trading_enabled": self.trading_enabled, "live": self.live,
                "mode": "LIVE" if self.live else "DRY-RUN", "start_equity": self.start_equity}


class RiskManager:
    def __init__(self, state: TradingState) -> None:
        self.state = state

    def check(self, broker: Broker, symbol: str, side: str, notional: float,
              confidence: float) -> Tuple[bool, str]:
        """Return (allowed, reason)."""
        if not self.state.trading_enabled:
            return False, "trading is halted (kill switch on)."
        if notional <= 0:
            return False, "order notional must be positive."
        if notional > settings.max_order_notional + 1e-6:
            return False, f"order ${notional:.2f} exceeds MAX_ORDER_NOTIONAL ${settings.max_order_notional:.2f}."
        if confidence < settings.order_confidence_min:
            return False, (f"confidence {confidence:.2f} below ORDER_CONFIDENCE_MIN "
                           f"{settings.order_confidence_min:.2f}.")
        if not is_market_open():
            return False, "US market is closed."

        # Daily loss guard.
        if settings.daily_loss_limit > 0 and self.state.start_equity:
            acc = broker.get_account()
            if acc and (self.state.start_equity - acc.equity) >= settings.daily_loss_limit:
                self.state.trading_enabled = False  # trip the kill switch
                return False, (f"daily loss limit ${settings.daily_loss_limit:.2f} hit "
                               f"— trading halted.")

        if side == BUY:
            positions = broker.get_positions()
            existing = next((p for p in positions if p.symbol.upper() == symbol.upper()), None)
            existing_value = existing.market_value if existing else 0.0
            if existing_value + notional > settings.max_position_notional + 1e-6:
                return False, (f"{symbol} position would exceed MAX_POSITION_NOTIONAL "
                               f"${settings.max_position_notional:.2f}.")
            if existing is None and len(positions) >= settings.max_open_positions:
                return False, f"MAX_OPEN_POSITIONS ({settings.max_open_positions}) reached."
        return True, "ok"
