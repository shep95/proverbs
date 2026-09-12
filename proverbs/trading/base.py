"""Broker abstraction — a common interface every broker adapter implements."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

BUY = "buy"
SELL = "sell"

# OrderResult statuses
SUBMITTED = "submitted"   # sent to the (real or paper) broker
DRY_RUN = "dry_run"       # intended order logged, NOT sent (LIVE_TRADING=false)
REJECTED = "rejected"     # blocked by a risk rule / market closed / kill switch
ERROR = "error"           # broker/network failure


@dataclass
class BrokerAccount:
    cash: float
    buying_power: float
    equity: float
    currency: str = "USD"

    def to_dict(self) -> dict:
        return {"cash": round(self.cash, 2), "buying_power": round(self.buying_power, 2),
                "equity": round(self.equity, 2), "currency": self.currency}


@dataclass
class BrokerPosition:
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealized_pl(self) -> float:
        return (self.current_price - self.avg_cost) * self.quantity

    def to_dict(self) -> dict:
        return {"symbol": self.symbol, "quantity": round(self.quantity, 6),
                "avg_cost": round(self.avg_cost, 4), "current_price": round(self.current_price, 4),
                "market_value": round(self.market_value, 2),
                "unrealized_pl": round(self.unrealized_pl, 2)}


@dataclass
class OrderResult:
    status: str
    symbol: str
    side: str
    notional: float = 0.0
    quantity: float = 0.0
    price: float = 0.0
    order_id: Optional[str] = None
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status in (SUBMITTED, DRY_RUN)

    def to_dict(self) -> dict:
        return {"status": self.status, "symbol": self.symbol, "side": self.side,
                "notional": round(self.notional, 2), "quantity": round(self.quantity, 6),
                "price": round(self.price, 4), "order_id": self.order_id, "message": self.message}


@dataclass
class Broker(ABC):
    """Base broker. Subclasses implement the account/position/order methods."""

    name: str = "broker"
    connected: bool = field(default=False)

    @abstractmethod
    def connect(self) -> bool:
        """Establish/verify a session. Returns True on success."""

    @abstractmethod
    def get_account(self) -> Optional[BrokerAccount]:
        ...

    @abstractmethod
    def get_positions(self) -> List[BrokerPosition]:
        ...

    @abstractmethod
    def get_price(self, symbol: str) -> Optional[float]:
        ...

    @abstractmethod
    def submit_order(self, symbol: str, side: str, notional: float, price: float) -> OrderResult:
        """Actually place the order. Called only after risk checks + LIVE gate pass."""

    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        symbol = symbol.upper()
        for p in self.get_positions():
            if p.symbol.upper() == symbol:
                return p
        return None
