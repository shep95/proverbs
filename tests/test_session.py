"""Paper-trading session tests — offline, with patched brains + price."""
import pytest

from proverbs.brains.cultural_ai import CulturalInsight
from proverbs.brains.fiscal_ai import FiscalInsight
from proverbs.persistence import db
from proverbs.persistence.models import Base
from proverbs.services.engine import engine
from proverbs.trading.manager import trading
from proverbs.trading.session import parse_duration, sessions


@pytest.fixture(autouse=True)
def _fresh_db():
    Base.metadata.drop_all(db.engine)
    db.init_db()
    yield


def test_parse_duration():
    assert parse_duration("7d") == 7 * 86400
    assert parse_duration("24h") == 24 * 3600
    assert parse_duration("90m") == 90 * 60
    assert parse_duration("1h30m") == 5400
    assert parse_duration("2w") == 2 * 604800
    assert parse_duration("2") == 2 * 3600      # bare number => hours
    assert parse_duration("nonsense") is None


def _bull_fiscal(sym):
    return FiscalInsight(symbol=sym, signal=0.9, confidence=0.9, last_price=100.0,
                         sharpe_ratio=1.0, var_95=-0.02, trend_slope=0.001, trend_r2=0.5,
                         mean_return=0.001, volatility=0.01, moving_averages={"MA20": 100.0}, model="test")


def _bull_cultural(sym, limit=20):
    return CulturalInsight(symbol=sym, index=0.7, positive=8, neutral=1, negative=1,
                           sample_size=10, headlines=["great earnings"])


def test_paper_session_lifecycle(monkeypatch):
    monkeypatch.setattr(engine.fiscal, "analyze", _bull_fiscal)
    monkeypatch.setattr(engine.cultural, "analyze", _bull_cultural)

    res = sessions.start(5000.0, "1h", created_by="u1", label="Q3 demo")
    assert res["status"] == "success"
    assert sessions.active() is True

    # The session switched trading to a fresh paper broker; give it a price.
    monkeypatch.setattr(trading.broker, "get_price", lambda s: 100.0)

    engine.run_cycle(["AAPL", "MSFT"])

    report = sessions.report()
    assert report["session"]["status"] == "running"
    assert report["performance"]["starting_capital"] == 5000.0
    assert report["trades"]["count"] >= 1
    assert len(report["equity_curve"]) >= 2  # start + at least one cycle
    assert report["performance"]["current_equity"] > 0

    stop = sessions.stop()
    assert stop["status"] == "success"
    assert sessions.active() is False
    assert sessions.report()["session"]["status"] in ("stopped", "completed")


def test_start_rejects_bad_input():
    assert sessions.start(0, "7d")["status"] == "error"
    assert sessions.start(1000, "banana")["status"] == "error"
