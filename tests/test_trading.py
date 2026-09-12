"""Trading-layer tests — paper broker, risk manager, and the LIVE gate. Offline."""
import pytest

from proverbs.persistence import db
from proverbs.persistence.models import Base
from proverbs.trading.base import BUY, SELL, DRY_RUN, REJECTED, SUBMITTED
from proverbs.trading.manager import TradingManager
from proverbs.trading.paper_broker import PaperBroker
from proverbs.trading.risk import RiskManager, TradingState


@pytest.fixture(autouse=True)
def _fresh_db():
    # Reset all tables so paper-broker state does not bleed between tests.
    Base.metadata.drop_all(db.engine)
    db.init_db()
    yield


def test_paper_broker_buy_and_sell():
    b = PaperBroker()
    b.connect()
    acc0 = b.get_account()
    assert acc0.cash > 0

    r = b.submit_order("AAPL", BUY, notional=500.0, price=100.0)
    assert r.status == SUBMITTED
    assert round(r.quantity, 4) == 5.0

    pos = b.get_position("AAPL")
    assert pos is not None and round(pos.quantity, 4) == 5.0
    assert round(pos.avg_cost, 2) == 100.0

    # Sell half.
    r2 = b.submit_order("AAPL", SELL, notional=250.0, price=100.0)
    assert r2.status == SUBMITTED
    pos2 = b.get_position("AAPL")
    assert round(pos2.quantity, 4) == 2.5


def test_paper_broker_rejects_oversell():
    b = PaperBroker()
    b.connect()
    r = b.submit_order("MSFT", SELL, notional=100.0, price=50.0)
    assert r.status == "error"  # nothing held


def test_risk_blocks_when_halted():
    state = TradingState()
    state.trading_enabled = False
    rm = RiskManager(state)
    b = PaperBroker(); b.connect()
    ok, reason = rm.check(b, "AAPL", BUY, 100.0, 0.9)
    assert ok is False and "halt" in reason.lower()


def test_risk_blocks_oversized_order():
    state = TradingState()
    rm = RiskManager(state)
    b = PaperBroker(); b.connect()
    ok, reason = rm.check(b, "AAPL", BUY, 999999.0, 0.9)
    assert ok is False and "MAX_ORDER_NOTIONAL" in reason


def test_risk_blocks_low_confidence():
    state = TradingState()
    rm = RiskManager(state)
    b = PaperBroker(); b.connect()
    ok, reason = rm.check(b, "AAPL", BUY, 100.0, 0.0)
    assert ok is False and "confidence" in reason.lower()


def test_manager_dry_run_does_not_submit(monkeypatch):
    tm = TradingManager()
    tm.init()
    monkeypatch.setattr(tm.broker, "get_price", lambda s: 100.0)
    tm.set_live(False)  # dry-run
    result = tm.manual_order("AAPL", "buy", 100.0)
    assert result.status == DRY_RUN
    # Nothing should have been bought.
    assert tm.broker.get_position("AAPL") is None


def test_manager_live_paper_submits(monkeypatch):
    tm = TradingManager()
    tm.init()
    monkeypatch.setattr(tm.broker, "get_price", lambda s: 100.0)
    tm.set_live(True)  # live, but paper broker -> safe
    result = tm.manual_order("AAPL", "buy", 100.0)
    assert result.status == SUBMITTED
    assert tm.broker.get_position("AAPL") is not None


def test_kill_switch_blocks_orders(monkeypatch):
    tm = TradingManager()
    tm.init()
    monkeypatch.setattr(tm.broker, "get_price", lambda s: 100.0)
    tm.set_live(True)
    tm.halt()
    result = tm.manual_order("AAPL", "buy", 100.0)
    assert result.status == REJECTED
