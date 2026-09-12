"""Paper broker — a persistent simulated brokerage backed by the app DB.

Distinct from the "balance-growth" gamified accounts in the Discord layer: this
is a real order-book simulation (cash + positions marked to market) so the
trading pipeline can be exercised end to end without touching real money.

Uses a dedicated account row (``broker_paper``) and the existing Position table.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from proverbs.config import settings
from proverbs.feeds.market_feed import latest_snapshot
from proverbs.persistence import db
from proverbs.persistence.models import Position
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

PAPER_ACCOUNT = "broker_paper"


class PaperBroker(Broker):
    def __init__(self) -> None:
        super().__init__(name="paper")

    def connect(self) -> bool:
        with db.session_scope() as s:
            acc = db.get_or_create_account(s, PAPER_ACCOUNT, "Paper Broker")
            # Seed starting cash once.
            if acc.total_deposited == 0 and acc.balance == 0:
                acc.balance = settings.paper_starting_cash
                acc.total_deposited = settings.paper_starting_cash
        self.connected = True
        logger.info("PaperBroker connected (starting cash $%.2f).", settings.paper_starting_cash)
        return True

    def get_price(self, symbol: str) -> Optional[float]:
        snap = latest_snapshot(symbol)
        return snap.last_price if snap else None

    def get_account(self) -> Optional[BrokerAccount]:
        with db.session_scope() as s:
            acc = db.get_or_create_account(s, PAPER_ACCOUNT, "Paper Broker")
            positions_value = 0.0
            for p in acc.positions:
                price = self.get_price(p.symbol) or p.current_price
                positions_value += p.quantity * price
            cash = acc.balance
            return BrokerAccount(cash=cash, buying_power=cash, equity=cash + positions_value)

    def get_positions(self) -> List[BrokerPosition]:
        out: List[BrokerPosition] = []
        with db.session_scope() as s:
            acc = db.get_or_create_account(s, PAPER_ACCOUNT, "Paper Broker")
            for p in acc.positions:
                if p.quantity <= 0:
                    continue
                price = self.get_price(p.symbol) or p.current_price
                p.current_price = price
                out.append(BrokerPosition(symbol=p.symbol, quantity=p.quantity,
                                          avg_cost=p.avg_cost, current_price=price))
        return out

    def submit_order(self, symbol: str, side: str, notional: float, price: float) -> OrderResult:
        symbol = symbol.upper()
        if price <= 0:
            return OrderResult(status=ERROR, symbol=symbol, side=side, message="No price available.")
        qty = notional / price
        with db.session_scope() as s:
            acc = db.get_or_create_account(s, PAPER_ACCOUNT, "Paper Broker")
            pos = s.query(Position).filter_by(account_id=acc.id, symbol=symbol).one_or_none()

            if side == BUY:
                if acc.balance < notional:
                    return OrderResult(status=ERROR, symbol=symbol, side=side,
                                       message=f"Insufficient paper cash (${acc.balance:.2f} < ${notional:.2f}).")
                acc.balance -= notional
                if pos is None:
                    pos = Position(account_id=acc.id, symbol=symbol, quantity=qty,
                                   avg_cost=price, current_price=price)
                    s.add(pos)
                else:
                    total_qty = pos.quantity + qty
                    pos.avg_cost = (pos.quantity * pos.avg_cost + qty * price) / total_qty if total_qty else price
                    pos.quantity = total_qty
                    pos.current_price = price
            else:  # SELL
                if pos is None or pos.quantity <= 0:
                    return OrderResult(status=ERROR, symbol=symbol, side=side,
                                       message="No position to sell.")
                sell_qty = min(qty, pos.quantity)
                proceeds = sell_qty * price
                acc.balance += proceeds
                pos.quantity -= sell_qty
                pos.current_price = price
                notional = proceeds
                qty = sell_qty
                if pos.quantity <= 1e-9:
                    s.delete(pos)

            db.record_transaction(s, acc, f"paper_{side}", notional,
                                  note=f"{side} {qty:.4f} {symbol} @ ${price:.2f}")
        return OrderResult(status=SUBMITTED, symbol=symbol, side=side, notional=notional,
                           quantity=qty, price=price, order_id="paper", message="Paper order filled.")
