"""Database engine, session factory, and account/transaction helpers."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator, List, Optional

from sqlalchemy import create_engine, desc
from sqlalchemy.orm import Session, sessionmaker

from proverbs.config import settings
from proverbs.persistence.models import Account, Base, Signal, Transaction

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
    # Ensure the shared global account exists.
    with session_scope() as s:
        get_or_create_account(s, GLOBAL_ACCOUNT_ID, "Global")


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
    """Persist an orchestrator Decision as a Signal row (audit trail)."""
    fiscal = decision.fiscal
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
