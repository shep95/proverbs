"""Unit tests for the three brains — all offline (no network)."""
import numpy as np
import pandas as pd

from proverbs.brains.cultural_ai import CulturalAI
from proverbs.brains.fiscal_ai import FiscalAI
from proverbs.brains.orchestrator import BUY, HOLD, REDUCE, Orchestrator
from proverbs.feeds.news_feed import Headline


def _headlines(titles):
    return [Headline(title=t, source="test") for t in titles]


def test_cultural_index_bounds_and_direction():
    ai = CulturalAI()
    bull = ai.score_headlines(_headlines([
        "Stocks soar to record highs on stellar earnings",
        "Investors thrilled as profits surge and outlook brightens",
    ]))
    bear = ai.score_headlines(_headlines([
        "Markets crash amid recession fears and massive losses",
        "Investors panic as stocks plunge to devastating lows",
    ]))
    assert -1.0 <= bull.index <= 1.0
    assert -1.0 <= bear.index <= 1.0
    assert bull.index > bear.index
    assert bull.label == "Bullish"


def test_cultural_empty_is_neutral():
    ai = CulturalAI()
    insight = ai.score_headlines([])
    assert insight.index == 0.0
    assert insight.sample_size == 0


def _synthetic_uptrend(n=400):
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    trend = np.linspace(100, 200, n)
    noise = np.random.default_rng(0).normal(0, 1.5, n)
    close = trend + noise
    return pd.DataFrame({
        "Open": close, "High": close + 1, "Low": close - 1,
        "Close": close, "Volume": np.random.default_rng(1).integers(1e6, 5e6, n),
    }, index=idx)


def test_fiscal_scores_uptrend():
    df = _synthetic_uptrend()
    insight = FiscalAI().score_dataframe("TEST", df)
    assert -1.0 <= insight.signal <= 1.0
    assert 0.0 <= insight.confidence <= 1.0
    assert insight.trend_slope > 0  # normalised slope positive for an uptrend
    assert "MA200" in insight.moving_averages
    assert insight.last_price > 0


def test_orchestrator_thresholds():
    orch = Orchestrator()

    class F:  # minimal fiscal stub
        signal = 0.9; confidence = 0.8; last_price = 100.0
        sharpe_ratio = 1.0; var_95 = -0.02

    class C:
        index = 0.8; sample_size = 10

    buy = orch.decide(F(), C(), "AAA")
    assert buy.action == BUY

    class Fn(F):
        signal = -0.9

    class Cn(C):
        index = -0.8

    reduce = orch.decide(Fn(), Cn(), "BBB")
    assert reduce.action == REDUCE

    class Fz(F):
        signal = 0.0

    class Cz(C):
        index = 0.0

    hold = orch.decide(Fz(), Cz(), "CCC")
    assert hold.action == HOLD


def test_auto_withdrawal_trigger():
    orch = Orchestrator()
    # deposit 100, balance 600 (>= 5.5x) -> withdraw 20% of 500 profit = 100
    res = orch.check_auto_withdrawal(100.0, 600.0)
    assert res.triggered is True
    assert round(res.amount, 2) == 100.0
    assert round(res.new_balance, 2) == 500.0

    # below the multiple -> no withdrawal
    res2 = orch.check_auto_withdrawal(100.0, 300.0)
    assert res2.triggered is False
    assert res2.amount == 0.0
