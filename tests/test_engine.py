"""Engine + persistence tests — offline, against a temp SQLite DB."""
import pytest

from proverbs.persistence import db
from proverbs.services.engine import engine


@pytest.fixture(autouse=True)
def _fresh_db():
    db.init_db()
    yield


def test_deposit_withdraw_flow():
    user = "user-1"
    r = engine.deposit(user, 1000.0, "Alice")
    assert r["status"] == "success"
    assert r["balance"] == 1000.0

    r = engine.withdraw(user, 250.0, "Alice")
    assert r["status"] == "success"
    assert r["balance"] == 750.0

    # over-withdraw rejected
    r = engine.withdraw(user, 10_000.0, "Alice")
    assert r["status"] == "error"


def test_negative_deposit_rejected():
    assert engine.deposit("user-2", -5.0)["status"] == "error"
    assert engine.withdraw("user-2", 0.0)["status"] == "error"


def test_set_risk_validation():
    assert engine.set_risk("user-3", "High")["status"] == "success"
    assert engine.set_risk("user-3", "high")["risk_level"] == "High"
    assert engine.set_risk("user-3", "bogus")["status"] == "error"


def test_history_records_transactions():
    user = "user-4"
    engine.deposit(user, 500.0, "Bob")
    engine.withdraw(user, 100.0, "Bob")
    history = engine.get_history(user)
    kinds = {h["kind"] for h in history}
    assert "deposit" in kinds and "withdrawal" in kinds


def test_account_isolation():
    engine.deposit("iso-a", 100.0)
    engine.deposit("iso-b", 300.0)
    assert engine.get_account("iso-a")["balance"] == 100.0
    assert engine.get_account("iso-b")["balance"] == 300.0
