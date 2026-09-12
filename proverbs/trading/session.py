"""Paper-trading sessions — a time-boxed live paper run with a shareholder report.

``/papertrade <amount> <duration>`` starts a session: the paper broker is reset to
``amount`` cash, live paper-trading is switched on, and the engine trades it every
cycle until ``duration`` elapses (or someone stops it). Equity is snapshotted each
cycle to build a performance report (return, drawdown, Sharpe, trades, positions,
equity curve) that can be shared with investors via Discord or the /report web page.

No real money is ever involved — sessions always run on the PaperBroker, even if
the deployment is otherwise configured for Robinhood.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import Optional

from proverbs.persistence import db
from proverbs.persistence.models import EquitySnapshot, PaperSession, Transaction
from proverbs.trading.manager import trading
from proverbs.trading.paper_broker import PAPER_ACCOUNT, PaperBroker

logger = logging.getLogger(__name__)

MIN_DURATION_SEC = 60           # 1 minute
MAX_DURATION_SEC = 90 * 86400   # 90 days


def parse_duration(text: str) -> Optional[int]:
    """Parse '7d', '24h', '90m', '2w', '1h30m' -> seconds. Bare number = hours."""
    if text is None:
        return None
    text = text.strip().lower()
    if not text:
        return None
    if text.isdigit():
        return int(text) * 3600  # bare number => hours
    units = {"w": 604800, "d": 86400, "h": 3600, "m": 60, "s": 1}
    matches = re.findall(r"(\d+)\s*([wdhms])", text)
    if not matches:
        return None
    total = sum(int(n) * units[u] for n, u in matches)
    return total or None


def human_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    mins, _ = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if mins and not days:
        parts.append(f"{mins}m")
    return " ".join(parts) or "0m"


class SessionManager:
    def __init__(self) -> None:
        self._prev_broker = None
        self._prev_live = None
        # A dedicated paper broker for reads/reports (independent of manager state).
        self._reader = PaperBroker()

    # --------------------------------------------------------------- helpers
    def _running_row(self, s) -> Optional[PaperSession]:
        return (s.query(PaperSession)
                .filter(PaperSession.status == "running")
                .order_by(PaperSession.id.desc())
                .first())

    def active(self) -> bool:
        with db.session_scope() as s:
            row = self._running_row(s)
            if row and time.time() < row.planned_end_ts:
                return True
        return False

    # ---------------------------------------------------------------- start
    def start(self, amount: float, duration_text: str, created_by: str = "",
              label: str = "") -> dict:
        if amount <= 0:
            return {"status": "error", "message": "Starting amount must be positive."}
        seconds = parse_duration(duration_text)
        if seconds is None:
            return {"status": "error", "message": "Couldn't parse duration. Try e.g. `7d`, `24h`, `90m`."}
        seconds = max(MIN_DURATION_SEC, min(MAX_DURATION_SEC, seconds))

        # Stop any session already running.
        self.stop(status="stopped", note="superseded by a new session")

        # Reset the paper broker to the requested capital and switch trading to it.
        pb = PaperBroker()
        pb.connect()
        pb.reset(amount)
        self._prev_broker = trading.broker
        self._prev_live = trading.state.live
        trading.broker = pb
        trading.state.live = True
        trading.state.trading_enabled = True
        trading.state.start_equity = amount

        now = time.time()
        with db.session_scope() as s:
            row = PaperSession(status="running", label=label, created_by=str(created_by),
                               starting_capital=amount, start_ts=now,
                               planned_end_ts=now + seconds, ended_ts=0.0, cycles=0)
            s.add(row)
            s.flush()
            s.add(EquitySnapshot(session_id=row.id, ts=now, equity=amount, cash=amount))
            sid = row.id
        logger.info("Paper session #%s started: $%.2f for %s.", sid, amount, human_duration(seconds))
        return {"status": "success", "session_id": sid, "starting_capital": amount,
                "duration": human_duration(seconds),
                "ends_at": datetime.fromtimestamp(now + seconds, tz=timezone.utc).isoformat()}

    # ----------------------------------------------------------- cycle hook
    def on_cycle(self) -> None:
        """Called at the end of each engine cycle: snapshot + auto-expire."""
        now = time.time()
        with db.session_scope() as s:
            row = self._running_row(s)
            if row is None:
                return
            acc = self._reader.get_account()
            equity = acc.equity if acc else row.starting_capital
            cash = acc.cash if acc else row.starting_capital
            s.add(EquitySnapshot(session_id=row.id, ts=now, equity=equity, cash=cash))
            row.cycles += 1
            expired = now >= row.planned_end_ts
        if expired:
            self.stop(status="completed", note="duration elapsed")

    # ----------------------------------------------------------------- stop
    def stop(self, status: str = "stopped", note: str = "") -> dict:
        with db.session_scope() as s:
            row = self._running_row(s)
            if row is None:
                return {"status": "noop", "message": "No running session."}
            acc = self._reader.get_account()
            row.final_equity = acc.equity if acc else row.starting_capital
            row.ended_ts = time.time()
            row.status = status
            sid = row.id
        # Restore the previous broker/live mode.
        if self._prev_broker is not None:
            trading.broker = self._prev_broker
        if self._prev_live is not None:
            trading.state.live = self._prev_live
        self._prev_broker = None
        self._prev_live = None
        logger.info("Paper session #%s %s (%s).", sid, status, note)
        return {"status": "success", "session_id": sid, "final_status": status}

    def resume_active(self) -> None:
        """On startup, re-attach the paper broker if a session is still running."""
        with db.session_scope() as s:
            row = self._running_row(s)
            if row is None:
                return
            if time.time() >= row.planned_end_ts:
                # Expired while we were down — close it out.
                acc = self._reader.get_account()
                row.final_equity = acc.equity if acc else row.starting_capital
                row.ended_ts = row.planned_end_ts
                row.status = "completed"
                logger.info("Paper session #%s completed (expired during downtime).", row.id)
                return
            sid = row.id
        pb = PaperBroker()
        pb.connect()
        self._prev_broker = trading.broker
        self._prev_live = trading.state.live
        trading.broker = pb
        trading.state.live = True
        trading.state.trading_enabled = True
        logger.info("Resumed running paper session #%s after restart.", sid)

    # --------------------------------------------------------------- report
    def report(self) -> Optional[dict]:
        with db.session_scope() as s:
            row = (s.query(PaperSession).order_by(PaperSession.id.desc()).first())
            if row is None:
                return None
            snapshots = [snap.to_dict() for snap in row.snapshots]
            # Lazily expire a running-but-past-due session.
            running = row.status == "running"
            if running and time.time() >= row.planned_end_ts:
                running = False  # report reflects completion; on_cycle/stop will persist
            session_info = {
                "id": row.id, "status": ("completed" if (not running and row.status == "running")
                                         else row.status),
                "label": row.label, "created_by": row.created_by,
                "starting_capital": round(row.starting_capital, 2),
                "start_ts": row.start_ts, "planned_end_ts": row.planned_end_ts,
                "ended_ts": row.ended_ts, "cycles": row.cycles,
            }
            start_dt = datetime.utcfromtimestamp(row.start_ts)
            # Count trades within this session window.
            buys = (s.query(Transaction)
                    .join(Transaction.account)
                    .filter(Transaction.kind == "paper_buy", Transaction.timestamp >= start_dt).count())
            sells = (s.query(Transaction)
                     .join(Transaction.account)
                     .filter(Transaction.kind == "paper_sell", Transaction.timestamp >= start_dt).count())
            # Closed-lot stats (win rate / best / worst / avg hold) for this session window.
            realized = db.realized_trades_since(s, row.start_ts)
            closed = len(realized)
            wins = sum(1 for t in realized if t.pnl > 0)
            best = max((t.pnl for t in realized), default=0.0)
            worst = min((t.pnl for t in realized), default=0.0)
            avg_hold = (sum(max(0.0, t.closed_ts - t.opened_ts) for t in realized) / closed) if closed else 0.0
            trade_stats = {
                "closed_trades": closed,
                "win_rate": round(wins / closed, 4) if closed else None,
                "wins": wins, "losses": closed - wins,
                "best_trade": round(best, 2), "worst_trade": round(worst, 2),
                "avg_hold": human_duration(avg_hold) if closed else "—",
            }
            # Signal-direction accuracy over the watchlist (all graded signals).
            acc_stats = db.accuracy_stats(s)

        # Live figures from the paper account.
        acc = self._reader.get_account()
        positions = [p.to_dict() for p in self._reader.get_positions()]
        current_equity = acc.equity if acc else session_info["starting_capital"]
        cash = acc.cash if acc else current_equity
        if session_info["status"] != "running":
            current_equity = row.final_equity or current_equity

        start_cap = session_info["starting_capital"] or 1.0
        pnl = current_equity - session_info["starting_capital"]
        return_pct = pnl / start_cap * 100.0

        # Equity-curve derived metrics.
        equities = [snap["equity"] for snap in snapshots] or [current_equity]
        peak = equities[0]
        max_dd = 0.0
        for e in equities:
            peak = max(peak, e)
            if peak > 0:
                max_dd = min(max_dd, (e - peak) / peak)
        rets = [(equities[i] - equities[i - 1]) / equities[i - 1]
                for i in range(1, len(equities)) if equities[i - 1] > 0]
        if len(rets) >= 2:
            mean = sum(rets) / len(rets)
            var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
            std = var ** 0.5
            sharpe = (mean / std) if std > 0 else 0.0
        else:
            sharpe = 0.0

        now = time.time()
        elapsed = min(now, session_info["planned_end_ts"]) - session_info["start_ts"]
        if session_info["ended_ts"]:
            elapsed = session_info["ended_ts"] - session_info["start_ts"]
        total = max(1.0, session_info["planned_end_ts"] - session_info["start_ts"])
        remaining = max(0.0, session_info["planned_end_ts"] - now) if session_info["status"] == "running" else 0.0

        return {
            "session": {
                **session_info,
                "start_iso": datetime.fromtimestamp(session_info["start_ts"], tz=timezone.utc).isoformat(),
                "planned_end_iso": datetime.fromtimestamp(session_info["planned_end_ts"], tz=timezone.utc).isoformat(),
                "ended_iso": (datetime.fromtimestamp(session_info["ended_ts"], tz=timezone.utc).isoformat()
                              if session_info["ended_ts"] else None),
                "elapsed_human": human_duration(elapsed),
                "remaining_human": human_duration(remaining),
                "progress_pct": round(min(100.0, elapsed / total * 100.0), 1),
            },
            "performance": {
                "starting_capital": round(session_info["starting_capital"], 2),
                "current_equity": round(current_equity, 2),
                "cash": round(cash, 2),
                "invested": round(current_equity - cash, 2),
                "pnl": round(pnl, 2),
                "return_pct": round(return_pct, 3),
                "max_drawdown_pct": round(max_dd * 100.0, 3),
                "sharpe_per_cycle": round(sharpe, 3),
                "peak_equity": round(peak, 2),
            },
            "trades": {"count": buys + sells, "buys": buys, "sells": sells, **trade_stats},
            "accuracy": acc_stats,
            "positions": positions,
            "equity_curve": snapshots,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def leaderboard(self, limit: int = 10) -> list:
        """Rank finished sessions by return %, best first (investor comparison)."""
        out = []
        with db.session_scope() as s:
            rows = (s.query(PaperSession)
                    .filter(PaperSession.status.in_(["completed", "stopped"]))
                    .all())
            for row in rows:
                start = row.starting_capital or 1.0
                final = row.final_equity or row.starting_capital
                out.append({
                    "session_id": row.id,
                    "label": row.label or f"session #{row.id}",
                    "created_by": row.created_by,
                    "starting_capital": round(row.starting_capital, 2),
                    "final_equity": round(final, 2),
                    "return_pct": round((final - row.starting_capital) / start * 100.0, 3),
                    "status": row.status,
                })
        out.sort(key=lambda x: x["return_pct"], reverse=True)
        return out[:limit]


sessions = SessionManager()
