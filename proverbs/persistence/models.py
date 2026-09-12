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

import json
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
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
    opened_ts: Mapped[float] = mapped_column(Float, default=0.0)  # epoch of first buy
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
    model: Mapped[str] = mapped_column(String(32), default="")
    proba_up: Mapped[float] = mapped_column(Float, default=0.5)

    # Fiscal feature vector (the inputs that produced the signal).
    rsi: Mapped[float] = mapped_column(Float, default=0.0)
    macd_hist: Mapped[float] = mapped_column(Float, default=0.0)
    bb_pos: Mapped[float] = mapped_column(Float, default=0.0)
    volume_z: Mapped[float] = mapped_column(Float, default=0.0)
    ma_ratio_20: Mapped[float] = mapped_column(Float, default=0.0)
    ma_ratio_50: Mapped[float] = mapped_column(Float, default=0.0)
    ma_ratio_200: Mapped[float] = mapped_column(Float, default=0.0)

    # Market context at signal time.
    volume: Mapped[float] = mapped_column(Float, default=0.0)
    day_high: Mapped[float] = mapped_column(Float, default=0.0)
    day_low: Mapped[float] = mapped_column(Float, default=0.0)
    week52_high: Mapped[float] = mapped_column(Float, default=0.0)
    week52_low: Mapped[float] = mapped_column(Float, default=0.0)

    # Cultural detail.
    cultural_positive: Mapped[int] = mapped_column(Integer, default=0)
    cultural_neutral: Mapped[int] = mapped_column(Integer, default=0)
    cultural_negative: Mapped[int] = mapped_column(Integer, default=0)
    cultural_sample: Mapped[int] = mapped_column(Integer, default=0)
    news_sources: Mapped[str] = mapped_column(String(256), default="")
    top_headlines: Mapped[str] = mapped_column(Text, default="")  # JSON list

    # Accuracy grading (filled once a later price is available).
    predicted_up: Mapped[int] = mapped_column(Integer, default=0)   # 1 = predicted up
    graded_ts: Mapped[float] = mapped_column(Float, default=0.0)    # 0 = ungraded
    outcome_price: Mapped[float] = mapped_column(Float, default=0.0)
    outcome_return: Mapped[float] = mapped_column(Float, default=0.0)
    correct: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # 1/0/None

    def to_dict(self) -> dict:
        headlines = []
        if self.top_headlines:
            try:
                headlines = json.loads(self.top_headlines)
            except Exception:
                headlines = []
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
            "model": self.model,
            "proba_up": round(self.proba_up, 4),
            "features": {
                "rsi": round(self.rsi, 2),
                "macd_hist": round(self.macd_hist, 6),
                "bb_pos": round(self.bb_pos, 4),
                "volume_z": round(self.volume_z, 4),
                "ma_ratio_20": round(self.ma_ratio_20, 4),
                "ma_ratio_50": round(self.ma_ratio_50, 4),
                "ma_ratio_200": round(self.ma_ratio_200, 4),
            },
            "market": {
                "volume": self.volume,
                "day_high": round(self.day_high, 4),
                "day_low": round(self.day_low, 4),
                "week52_high": round(self.week52_high, 4),
                "week52_low": round(self.week52_low, 4),
            },
            "cultural": {
                "positive": self.cultural_positive,
                "neutral": self.cultural_neutral,
                "negative": self.cultural_negative,
                "sample": self.cultural_sample,
                "sources": self.news_sources,
                "headlines": headlines,
            },
            "grading": {
                "graded": bool(self.graded_ts),
                "predicted_up": bool(self.predicted_up),
                "outcome_price": round(self.outcome_price, 4),
                "outcome_return": round(self.outcome_return, 6),
                "correct": (None if self.correct is None else bool(self.correct)),
            },
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


class WatchlistItem(Base):
    """Runtime-mutable watchlist (seeded from the WATCHLIST env var on first run)."""
    __tablename__ = "watchlist"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    added_by: Mapped[str] = mapped_column(String(64), default="")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class Alert(Base):
    """A per-user score threshold alert (fires once, then deactivates)."""
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    channel_id: Mapped[str] = mapped_column(String(64), default="")
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    direction: Mapped[str] = mapped_column(String(8))  # above | below
    threshold: Mapped[float] = mapped_column(Float, default=0.0)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    fired_ts: Mapped[float] = mapped_column(Float, default=0.0)

    def to_dict(self) -> dict:
        return {"id": self.id, "user_id": self.user_id, "symbol": self.symbol,
                "direction": self.direction, "threshold": round(self.threshold, 4),
                "active": self.active}


class RealizedTrade(Base):
    """A closed (sold) paper lot, for win-rate / best-worst / hold-time reporting."""
    __tablename__ = "realized_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    quantity: Mapped[float] = mapped_column(Float, default=0.0)
    cost_basis: Mapped[float] = mapped_column(Float, default=0.0)
    proceeds: Mapped[float] = mapped_column(Float, default=0.0)
    pnl: Mapped[float] = mapped_column(Float, default=0.0)
    opened_ts: Mapped[float] = mapped_column(Float, default=0.0)
    closed_ts: Mapped[float] = mapped_column(Float, default=0.0)

    def to_dict(self) -> dict:
        return {"symbol": self.symbol, "quantity": round(self.quantity, 6),
                "cost_basis": round(self.cost_basis, 2), "proceeds": round(self.proceeds, 2),
                "pnl": round(self.pnl, 2), "opened_ts": self.opened_ts, "closed_ts": self.closed_ts,
                "hold_seconds": max(0.0, self.closed_ts - self.opened_ts)}
