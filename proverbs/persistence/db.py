"""Database engine, session factory, and account/transaction helpers."""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from typing import Iterator, List, Optional

from sqlalchemy import create_engine, desc
from sqlalchemy.orm import Session, sessionmaker

from proverbs.config import settings
from proverbs.persistence.models import (
    Account,
    Alert,
    Base,
    RealizedTrade,
    Signal,
    Transaction,
    WatchlistItem,
)

logger = logging.getLogger(__name__)

GLOBAL_ACCOUNT_ID = "global"  # the shared account used by the dashboard/engine

_connect_args = {}
if settings.database_url.startswith("sqlite"):
    # Allow use across the Flask thread + asyncio bot thread.
    _connect_args = {"check_same_thread": False}

engine = create_engine(settings.database_url, echo=False, future=True,
                       pool_pre_ping=True, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, class_=Session, future=True)


def init_db() -> None:
    Base.metadata.create_all(engine)
    logger.info("Database initialised at %s", settings.database_url.split("@")[-1])
    # Ensure the shared global account exists, and seed the watchlist once.
    with session_scope() as s:
        get_or_create_account(s, GLOBAL_ACCOUNT_ID, "Global")
        if s.query(WatchlistItem).count() == 0:
            for sym in settings.watchlist:
                s.add(WatchlistItem(symbol=sym.upper(), added_by="env", active=True))


@contextmanager
def session_scope() -> Iterator[Session]:
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def get_or_create_account(s: Session, discord_user_id: str, display_name: str = "") -> Account:
    acc = s.query(Account).filter_by(discord_user_id=str(discord_user_id)).one_or_none()
    if acc is None:
        acc = Account(discord_user_id=str(discord_user_id), display_name=display_name or str(discord_user_id))
        s.add(acc)
        s.flush()
    elif display_name and acc.display_name != display_name:
        acc.display_name = display_name
    return acc


def record_transaction(s: Session, account: Account, kind: str, amount: float, note: str = "") -> Transaction:
    txn = Transaction(account_id=account.id, kind=kind, amount=amount,
                      balance_after=account.balance, note=note)
    s.add(txn)
    return txn


def save_signal(s: Session, decision) -> Signal:
    """Persist an orchestrator Decision as a Signal row (full audit trail)."""
    fiscal = decision.fiscal
    cultural = decision.cultural
    feats = (fiscal.features if fiscal else {}) or {}
    proba_up = fiscal.proba_up if fiscal else 0.5
    headlines = cultural.headlines if cultural else []
    sources = ", ".join(cultural.sources) if cultural and cultural.sources else ""

    sig = Signal(
        symbol=decision.symbol,
        fiscal_score=decision.fiscal_signal,
        cultural_score=decision.cultural_signal,
        combined_score=decision.combined_score,
        confidence=decision.confidence,
        action=decision.action,
        last_price=fiscal.last_price if fiscal else 0.0,
        sharpe_ratio=fiscal.sharpe_ratio if fiscal else 0.0,
        var_95=fiscal.var_95 if fiscal else 0.0,
        model=fiscal.model if fiscal else "",
        proba_up=proba_up,
        rsi=feats.get("rsi", 0.0),
        macd_hist=feats.get("macd_hist", 0.0),
        bb_pos=feats.get("bb_pos", 0.0),
        volume_z=feats.get("volume_z", 0.0),
        ma_ratio_20=feats.get("ma_ratio_20", 0.0),
        ma_ratio_50=feats.get("ma_ratio_50", 0.0),
        ma_ratio_200=feats.get("ma_ratio_200", 0.0),
        volume=fiscal.volume if fiscal else 0.0,
        day_high=fiscal.day_high if fiscal else 0.0,
        day_low=fiscal.day_low if fiscal else 0.0,
        week52_high=fiscal.week52_high if fiscal else 0.0,
        week52_low=fiscal.week52_low if fiscal else 0.0,
        cultural_positive=cultural.positive if cultural else 0,
        cultural_neutral=cultural.neutral if cultural else 0,
        cultural_negative=cultural.negative if cultural else 0,
        cultural_sample=cultural.sample_size if cultural else 0,
        news_sources=sources[:256],
        top_headlines=json.dumps(headlines[:5]),
        predicted_up=1 if proba_up >= 0.5 else 0,
    )
    s.add(sig)
    return sig


def recent_signals(s: Session, symbol: Optional[str] = None, limit: int = 20) -> List[Signal]:
    q = s.query(Signal)
    if symbol:
        q = q.filter(Signal.symbol == symbol.upper())
    return q.order_by(desc(Signal.timestamp)).limit(limit).all()


def latest_signal_per_symbol(s: Session) -> List[Signal]:
    """Most recent signal for each symbol (small watchlists — simple approach)."""
    symbols = [row[0] for row in s.query(Signal.symbol).distinct().all()]
    out: List[Signal] = []
    for sym in symbols:
        sig = s.query(Signal).filter(Signal.symbol == sym).order_by(desc(Signal.timestamp)).first()
        if sig:
            out.append(sig)
    return out


def recent_transactions(s: Session, account: Account, limit: int = 20) -> List[Transaction]:
    return (s.query(Transaction)
            .filter(Transaction.account_id == account.id)
            .order_by(desc(Transaction.timestamp))
            .limit(limit)
            .all())


# ------------------------------------------------------------------ watchlist
def get_watchlist(s: Session) -> List[str]:
    rows = s.query(WatchlistItem).filter(WatchlistItem.active.is_(True)).order_by(WatchlistItem.symbol).all()
    symbols = [r.symbol for r in rows]
    return symbols or [x.upper() for x in settings.watchlist]


def add_watch(s: Session, symbol: str, added_by: str = "") -> bool:
    symbol = symbol.upper().strip()
    if not symbol:
        return False
    row = s.query(WatchlistItem).filter_by(symbol=symbol).one_or_none()
    if row is None:
        s.add(WatchlistItem(symbol=symbol, added_by=str(added_by), active=True))
        return True
    if not row.active:
        row.active = True
        return True
    return False  # already present


def remove_watch(s: Session, symbol: str) -> bool:
    symbol = symbol.upper().strip()
    row = s.query(WatchlistItem).filter_by(symbol=symbol).one_or_none()
    if row is None or not row.active:
        return False
    row.active = False
    return True


# --------------------------------------------------------------------- alerts
def add_alert(s: Session, user_id: str, channel_id: str, symbol: str,
              direction: str, threshold: float) -> Alert:
    alert = Alert(user_id=str(user_id), channel_id=str(channel_id), symbol=symbol.upper(),
                  direction=direction.lower(), threshold=threshold, active=True)
    s.add(alert)
    s.flush()
    return alert


def list_alerts(s: Session, user_id: str) -> List[Alert]:
    return (s.query(Alert)
            .filter(Alert.user_id == str(user_id), Alert.active.is_(True))
            .order_by(Alert.id).all())


def active_alerts(s: Session) -> List[Alert]:
    return s.query(Alert).filter(Alert.active.is_(True)).all()


# ------------------------------------------------------------------- grading
def ungraded_signals(s: Session, symbol: str, before_ts_iso) -> List[Signal]:
    """Signals for a symbol that are ungraded and older than a cutoff datetime."""
    return (s.query(Signal)
            .filter(Signal.symbol == symbol.upper(),
                    Signal.graded_ts == 0.0,
                    Signal.last_price > 0.0,
                    Signal.timestamp <= before_ts_iso)
            .all())


def accuracy_stats(s: Session, symbol: Optional[str] = None) -> dict:
    q = s.query(Signal).filter(Signal.graded_ts > 0.0, Signal.correct.isnot(None))
    if symbol:
        q = q.filter(Signal.symbol == symbol.upper())
    graded = q.all()
    total = len(graded)
    correct = sum(1 for g in graded if g.correct == 1)
    return {
        "graded": total,
        "correct": correct,
        "hit_rate": round(correct / total, 4) if total else None,
        "symbol": symbol.upper() if symbol else "ALL",
    }


# ----------------------------------------------------------- realized trades
def record_realized_trade(s: Session, symbol: str, quantity: float, cost_basis: float,
                          proceeds: float, opened_ts: float, closed_ts: float,
                          session_id: Optional[int] = None) -> RealizedTrade:
    trade = RealizedTrade(session_id=session_id, symbol=symbol.upper(), quantity=quantity,
                          cost_basis=cost_basis, proceeds=proceeds, pnl=proceeds - cost_basis,
                          opened_ts=opened_ts, closed_ts=closed_ts)
    s.add(trade)
    return trade


def realized_trades_since(s: Session, since_ts: float) -> List[RealizedTrade]:
    return (s.query(RealizedTrade)
            .filter(RealizedTrade.closed_ts >= since_ts)
            .order_by(RealizedTrade.closed_ts).all())
