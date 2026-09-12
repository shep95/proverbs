"""SQLAlchemy ORM models.

Phase 5 of the rebuild: replace the global python variables (user_balance,
user_deposit, ...) that reset on every restart with real tables. On Railway,
attach a Postgres plugin and DATABASE_URL is used automatically; otherwise a
local SQLite file is used.

Tables:
  account       — one row per Discord user (balance, deposits, risk level)
  positions     — simulated holdings per account/symbol
  signals       — every orchestrator decision (audit trail)
  transactions  — deposits / withdrawals / auto-withdrawals
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Account(Base):
    __tablename__ = "account"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Discord user id as string; "global" for the shared/dashboard account.
    discord_user_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(128), default="")
    balance: Mapped[float] = mapped_column(Float, default=0.0)
    total_deposited: Mapped[float] = mapped_column(Float, default=0.0)
    total_withdrawn: Mapped[float] = mapped_column(Float, default=0.0)
    risk_level: Mapped[str] = mapped_column(String(16), default="Medium")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)

    positions: Mapped[list["Position"]] = relationship(back_populates="account",
                                                        cascade="all, delete-orphan")
    transactions: Mapped[list["Transaction"]] = relationship(back_populates="account",
                                                             cascade="all, delete-orphan")

    def to_dict(self) -> dict:
        return {
            "discord_user_id": self.discord_user_id,
            "display_name": self.display_name,
            "balance": round(self.balance, 2),
            "total_deposited": round(self.total_deposited, 2),
            "total_withdrawn": round(self.total_withdrawn, 2),
            "risk_level": self.risk_level,
            "profit": round(self.balance - (self.total_deposited - self.total_withdrawn), 2),
        }


class Position(Base):
    __tablename__ = "positions"
    __table_args__ = (UniqueConstraint("account_id", "symbol", name="uq_account_symbol"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("account.id"), index=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    quantity: Mapped[float] = mapped_column(Float, default=0.0)
    avg_cost: Mapped[float] = mapped_column(Float, default=0.0)
    current_price: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)

    account: Mapped["Account"] = relationship(back_populates="positions")

    def to_dict(self) -> dict:
        market_value = self.quantity * self.current_price
        return {
            "symbol": self.symbol,
            "quantity": round(self.quantity, 4),
            "avg_cost": round(self.avg_cost, 4),
            "current_price": round(self.current_price, 4),
            "market_value": round(market_value, 2),
        }


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    fiscal_score: Mapped[float] = mapped_column(Float, default=0.0)
    cultural_score: Mapped[float] = mapped_column(Float, default=0.0)
    combined_score: Mapped[float] = mapped_column(Float, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    action: Mapped[str] = mapped_column(String(16), default="HOLD")
    last_price: Mapped[float] = mapped_column(Float, default=0.0)
    sharpe_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    var_95: Mapped[float] = mapped_column(Float, default=0.0)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "fiscal_score": round(self.fiscal_score, 4),
            "cultural_score": round(self.cultural_score, 4),
            "combined_score": round(self.combined_score, 4),
            "confidence": round(self.confidence, 4),
            "action": self.action,
            "last_price": round(self.last_price, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "var_95": round(self.var_95, 6),
        }


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("account.id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    kind: Mapped[str] = mapped_column(String(24))  # deposit | withdrawal | auto_withdrawal
    amount: Mapped[float] = mapped_column(Float, default=0.0)
    balance_after: Mapped[float] = mapped_column(Float, default=0.0)
    note: Mapped[str] = mapped_column(String(256), default="")

    account: Mapped["Account"] = relationship(back_populates="transactions")

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "kind": self.kind,
            "amount": round(self.amount, 2),
            "balance_after": round(self.balance_after, 2),
            "note": self.note,
        }


class PaperSession(Base):
    """A time-boxed live paper-trading run, tracked for a shareholder report."""
    __tablename__ = "paper_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    status: Mapped[str] = mapped_column(String(16), default="running", index=True)  # running|completed|stopped
    label: Mapped[str] = mapped_column(String(128), default="")
    created_by: Mapped[str] = mapped_column(String(64), default="")
    starting_capital: Mapped[float] = mapped_column(Float, default=0.0)
    start_ts: Mapped[float] = mapped_column(Float, default=0.0)          # epoch seconds
    planned_end_ts: Mapped[float] = mapped_column(Float, default=0.0)    # epoch seconds
    ended_ts: Mapped[float] = mapped_column(Float, default=0.0)          # 0 while running
    final_equity: Mapped[float] = mapped_column(Float, default=0.0)
    cycles: Mapped[int] = mapped_column(Integer, default=0)

    snapshots: Mapped[list["EquitySnapshot"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="EquitySnapshot.ts")


class EquitySnapshot(Base):
    """A point on a session's equity curve (recorded each cycle)."""
    __tablename__ = "equity_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("paper_sessions.id"), index=True)
    ts: Mapped[float] = mapped_column(Float, default=0.0)  # epoch seconds
    equity: Mapped[float] = mapped_column(Float, default=0.0)
    cash: Mapped[float] = mapped_column(Float, default=0.0)

    session: Mapped["PaperSession"] = relationship(back_populates="snapshots")

    def to_dict(self) -> dict:
        return {"ts": self.ts, "equity": round(self.equity, 2), "cash": round(self.cash, 2)}
