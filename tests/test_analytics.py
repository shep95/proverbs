"""Tests for signal-accuracy grading, watchlist, weights, alerts, realized trades."""
import time

import pytest

from proverbs.brains.cultural_ai import CulturalInsight
from proverbs.brains.fiscal_ai import FiscalInsight
from proverbs.persistence import db
from proverbs.persistence.models import Base, Signal
from proverbs.services.engine import engine
from proverbs.trading.manager import trading
from proverbs.trading.session import sessions


@pytest.fixture(autouse=True)
def _fresh_db():
    Base.metadata.drop_all(db.engine)
    db.init_db()
    yield


def test_watchlist_add_remove():
    engine.add_to_watchlist("NVDA", "u1")
    assert "NVDA" in engine.watchlist()
    assert engine.remove_from_watchlist("NVDA") is True
    assert "NVDA" not in engine.watchlist()
    # default seed present
    assert "AAPL" in engine.watchlist()


def test_set_weights_runtime():
    res = engine.set_weights(0.0, 1.0)
    assert res["status"] == "success"
    assert res["cultural_weight"] == 1.0 and res["fiscal_weight"] == 0.0
    assert engine.set_weights(-1, 0)["status"] == "error"
    engine.set_weights(0.65, 0.35)  # restore


def test_alerts_add_and_list():
    res = engine.add_alert("u1", "chan1", "AAPL", "above", 0.5)
    assert res["status"] == "success"
    rows = engine.list_alerts("u1")
    assert len(rows) == 1 and rows[0]["symbol"] == "AAPL"
    assert engine.add_alert("u1", "c", "AAPL", "sideways", 0.5)["status"] == "error"


def test_alert_fires_and_deactivates():
    engine.add_alert("u1", "chan1", "AAPL", "above", 0.5)
    triggered = engine._evaluate_alerts({"AAPL": 0.7})
    assert len(triggered) == 1 and triggered[0]["symbol"] == "AAPL"
    # one-shot: now inactive
    assert engine.list_alerts("u1") == []
    # below threshold should not fire
    engine.add_alert("u2", "chan1", "MSFT", "above", 0.9)
    assert engine._evaluate_alerts({"MSFT": 0.1}) == []


def test_signal_grading():
    # Insert a signal predicting UP, dated in the past, priced at 100.
    with db.session_scope() as s:
        sig = Signal(symbol="AAPL", combined_score=0.5, confidence=0.8, action="BUY",
                     last_price=100.0, predicted_up=1, proba_up=0.8)
        s.add(sig)
        s.flush()
        # Backdate it beyond the grading horizon.
        from datetime import datetime, timedelta
        sig.timestamp = datetime.utcnow() - timedelta(hours=48)

    graded = engine._grade_signals({"AAPL": 110.0})  # price rose -> prediction correct
    assert graded == 1
    stats = engine.accuracy_stats("AAPL")
    assert stats["graded"] == 1 and stats["correct"] == 1 and stats["hit_rate"] == 1.0


def test_realized_trade_and_win_rate(monkeypatch):
    # Start a session and force a buy then a sell to realize a profit.
    sessions.start(10000.0, "1h", created_by="u1")
    monkeypatch.setattr(trading.broker, "get_price", lambda s: 100.0)
    trading.manual_order("AAPL", "buy", 500.0)      # 5 shares @ 100
    monkeypatch.setattr(trading.broker, "get_price", lambda s: 120.0)
    trading.manual_order("AAPL", "sell", 240.0)     # sell 2 shares @ 120 -> +40 pnl
    report = sessions.report()
    assert report["trades"]["closed_trades"] >= 1
    assert report["trades"]["win_rate"] == 1.0
    assert report["trades"]["best_trade"] > 0
    sessions.stop()


def test_leaderboard_ranks_sessions():
    r1 = sessions.start(1000.0, "1m", created_by="u1", label="A")
    sessions.stop()
    r2 = sessions.start(1000.0, "1m", created_by="u2", label="B")
    sessions.stop()
    board = sessions.leaderboard()
    assert len(board) >= 2
    # sorted by return desc
    assert board[0]["return_pct"] >= board[-1]["return_pct"]


def test_user_leaderboard_aggregates_by_user():
    sessions.start(1000.0, "1m", created_by="u1", label="A1"); sessions.stop()
    sessions.start(1000.0, "1m", created_by="u1", label="A2"); sessions.stop()
    sessions.start(1000.0, "1m", created_by="u2", label="B1"); sessions.stop()
    board = sessions.user_leaderboard()
    users = {row["user_id"]: row for row in board}
    assert "u1" in users and users["u1"]["sessions"] == 2
    assert "u2" in users and users["u2"]["sessions"] == 1
    # sorted by avg return desc
    assert board[0]["avg_return_pct"] >= board[-1]["avg_return_pct"]


def test_per_symbol_accuracy_horizon(monkeypatch):
    from proverbs.config import settings
    # Global 24h, but TSLA overridden to 1h.
    monkeypatch.setattr(settings, "accuracy_horizon_overrides", {"TSLA": 1.0})
    assert settings.horizon_hours_for("TSLA") == 1.0
    assert settings.horizon_hours_for("AAPL") == settings.accuracy_horizon_hours

    from datetime import datetime, timedelta
    with db.session_scope() as s:
        sig = Signal(symbol="TSLA", combined_score=0.4, confidence=0.7, action="BUY",
                     last_price=200.0, predicted_up=1, proba_up=0.7)
        s.add(sig)
        s.flush()
        sig.timestamp = datetime.utcnow() - timedelta(hours=2)  # older than 1h, younger than 24h
    # With the 1h override, a 2h-old TSLA signal is now gradeable.
    graded = engine._grade_signals({"TSLA": 210.0})
    assert graded == 1
