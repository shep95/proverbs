"""Robinhood broker adapter (equities/ETFs) via the unofficial ``robin_stocks``.

⚠️  Robinhood has no official public stock-trading API. ``robin_stocks`` talks to
    Robinhood's private endpoints — this is against their ToS and can break or get
    an account flagged. Use at your own risk. Credentials come from the environment
    only (never hardcoded), and headless login requires a TOTP 2FA shared secret.

Order placement here is only ever reached when LIVE_TRADING=true AND all risk
checks pass (see trading/manager.py). In dry-run the manager never calls
``submit_order`` on this class.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from proverbs.config import settings
from proverbs.trading.base import (
    BUY,
    SUBMITTED,
    ERROR,
    Broker,
    BrokerAccount,
    BrokerPosition,
    OrderResult,
)

logger = logging.getLogger(__name__)


class RobinhoodBroker(Broker):
    def __init__(self) -> None:
        super().__init__(name="robinhood")
        self._rh = None  # the robin_stocks.robinhood module

    # ------------------------------------------------------------------ auth
    def connect(self) -> bool:
        if not (settings.robinhood_username and settings.robinhood_password):
            logger.error("RobinhoodBroker: username/password not configured.")
            return False
        try:
            import robin_stocks.robinhood as rh  # lazy: only needed for this broker
        except Exception as exc:
            logger.error("RobinhoodBroker: robin_stocks not installed (%s). "
                         "Add it to requirements to use BROKER=robinhood.", exc)
            return False

        mfa_code = None
        if settings.robinhood_mfa_secret:
            try:
                import pyotp

                mfa_code = pyotp.TOTP(settings.robinhood_mfa_secret).now()
            except Exception as exc:
                logger.error("RobinhoodBroker: failed to generate TOTP code (%s).", exc)
                return False

        try:
            rh.authentication.login(
                username=settings.robinhood_username,
                password=settings.robinhood_password,
                mfa_code=mfa_code,
                store_session=True,
                expiresIn=86400,
            )
        except Exception as exc:
            logger.error("RobinhoodBroker: login failed (%s).", exc)
            return False

        self._rh = rh
        self.connected = True
        logger.info("RobinhoodBroker connected.")
        return True

    def _ensure(self) -> bool:
        return self.connected and self._rh is not None

    # -------------------------------------------------------------- read-only
    def get_price(self, symbol: str) -> Optional[float]:
        if not self._ensure():
            return None
        try:
            prices = self._rh.stocks.get_latest_price(symbol.upper())
            if prices and prices[0] is not None:
                return float(prices[0])
        except Exception as exc:
            logger.warning("RobinhoodBroker: get_price(%s) failed: %s", symbol, exc)
        return None

    def get_account(self) -> Optional[BrokerAccount]:
        if not self._ensure():
            return None
        try:
            profile = self._rh.profiles.load_account_profile()
            cash = float(profile.get("cash") or profile.get("portfolio_cash") or 0.0)
            buying_power = float(profile.get("buying_power") or cash)
            equity = cash
            positions = self.get_positions()
            equity += sum(p.market_value for p in positions)
            return BrokerAccount(cash=cash, buying_power=buying_power, equity=equity)
        except Exception as exc:
            logger.warning("RobinhoodBroker: get_account failed: %s", exc)
            return None

    def get_positions(self) -> List[BrokerPosition]:
        if not self._ensure():
            return []
        out: List[BrokerPosition] = []
        try:
            holdings = self._rh.account.build_holdings() or {}
            for symbol, h in holdings.items():
                qty = float(h.get("quantity", 0) or 0)
                if qty <= 0:
                    continue
                out.append(BrokerPosition(
                    symbol=symbol.upper(),
                    quantity=qty,
                    avg_cost=float(h.get("average_buy_price", 0) or 0),
                    current_price=float(h.get("price", 0) or 0),
                ))
        except Exception as exc:
            logger.warning("RobinhoodBroker: get_positions failed: %s", exc)
        return out

    # ------------------------------------------------------------- execution
    def submit_order(self, symbol: str, side: str, notional: float, price: float) -> OrderResult:
        if not self._ensure():
            return OrderResult(status=ERROR, symbol=symbol, side=side,
                               message="Robinhood session not connected.")
        symbol = symbol.upper()
        try:
            if side == BUY:
                resp = self._rh.orders.order_buy_fractional_by_price(symbol, round(notional, 2))
            else:
                resp = self._rh.orders.order_sell_fractional_by_price(symbol, round(notional, 2))
        except Exception as exc:
            logger.exception("RobinhoodBroker: order failed for %s", symbol)
            return OrderResult(status=ERROR, symbol=symbol, side=side, message=str(exc))

        if not resp or resp.get("detail") and not resp.get("id"):
            return OrderResult(status=ERROR, symbol=symbol, side=side,
                               notional=notional, price=price,
                               message=str(resp.get("detail") if resp else "empty response"))
        return OrderResult(status=SUBMITTED, symbol=symbol, side=side, notional=notional,
                           quantity=(notional / price if price else 0.0), price=price,
                           order_id=resp.get("id"), message="Order submitted to Robinhood.")
