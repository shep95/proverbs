"""Fiscal AI — the quantitative risk/reward brain (named for Edward Thorp).

Phase 3 of the rebuild: keep the correct Sharpe/VaR math from the original, but
replace the bare linear regression with a **feature-engineered model**:

  Features: 20/50/200-day MA (as price ratios), RSI, MACD, volume z-score,
            Bollinger band position, daily return, rolling volatility.
  Model:    scikit-learn HistGradientBoosting classifier predicting whether the
            next day closes up, with a probability -> confidence + directional
            signal in [-1, 1]. Falls back to a trend/momentum heuristic when
            there isn't enough history to train.

Output (FiscalInsight): directional signal + confidence, plus Sharpe ratio,
95% VaR, trend slope/R^2 and moving averages — all surfaced to the orchestrator.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from proverbs.feeds.market_feed import fetch_ohlcv

logger = logging.getLogger(__name__)


@dataclass
class FiscalInsight:
    symbol: str
    signal: float               # directional signal in [-1, 1]
    confidence: float           # model/heuristic confidence in [0, 1]
    last_price: float
    sharpe_ratio: float
    var_95: float               # 95% daily Value at Risk (negative number)
    trend_slope: float
    trend_r2: float
    mean_return: float
    volatility: float
    moving_averages: Dict[str, float] = field(default_factory=dict)
    model: str = "heuristic"
    # Feature vector (the inputs behind the signal) + raw model probability.
    features: Dict[str, float] = field(default_factory=dict)
    proba_up: float = 0.5
    # Market context.
    volume: float = 0.0
    day_high: float = 0.0
    day_low: float = 0.0
    week52_high: float = 0.0
    week52_low: float = 0.0

    @property
    def label(self) -> str:
        if self.signal > 0.15:
            return "Bullish"
        if self.signal < -0.15:
            return "Bearish"
        return "Neutral"


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(0.0, index=df.index)

    feat = pd.DataFrame(index=df.index)
    feat["ret"] = close.pct_change()
    feat["vol"] = feat["ret"].rolling(20).std()
    for w in (20, 50, 200):
        feat[f"ma_ratio_{w}"] = close / close.rolling(w).mean() - 1.0
    feat["rsi"] = _rsi(close)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal_line = macd.ewm(span=9, adjust=False).mean()
    feat["macd_hist"] = (macd - signal_line) / close
    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std().replace(0, np.nan)
    feat["vol_z"] = (volume - vol_mean) / vol_std
    ma20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std().replace(0, np.nan)
    feat["bb_pos"] = (close - ma20) / (2 * sd20)  # ~[-1,1] inside the bands
    return feat


def _risk_metrics(close: pd.Series) -> Dict[str, float]:
    returns = close.pct_change().dropna()
    if returns.empty:
        return {"mean_return": 0.0, "volatility": 0.0, "sharpe": 0.0, "var_95": 0.0}
    mean_r = float(returns.mean())
    std_r = float(returns.std())
    sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0.0  # annualised
    var_95 = float(np.percentile(returns, 5))
    return {"mean_return": mean_r, "volatility": std_r, "sharpe": sharpe, "var_95": var_95}


def _trend(close: pd.Series) -> Dict[str, float]:
    y = close.dropna().values
    if len(y) < 3:
        return {"slope": 0.0, "r2": 0.0}
    x = np.arange(len(y))
    try:
        from scipy.stats import linregress

        res = linregress(x, y)
        # Normalise slope by price so it is comparable across tickers.
        norm_slope = float(res.slope) / (float(np.mean(y)) or 1.0)
        return {"slope": norm_slope, "r2": float(res.rvalue) ** 2}
    except Exception:
        coeffs = np.polyfit(x, y, 1)
        return {"slope": float(coeffs[0]) / (float(np.mean(y)) or 1.0), "r2": 0.0}


def _model_signal(feat: pd.DataFrame, close: pd.Series) -> Optional[Dict[str, float]]:
    """Train a classifier on engineered features; predict next-day direction.

    Returns None when there isn't enough clean history to train.
    """
    label = (close.shift(-1) > close).astype(int)
    data = feat.copy()
    data["label"] = label
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 60:
        return None

    feature_cols = [c for c in data.columns if c != "label"]
    X = data[feature_cols].values
    y = data["label"].values
    if len(np.unique(y)) < 2:
        return None

    try:
        from sklearn.ensemble import HistGradientBoostingClassifier

        # Train on all but the final row; predict the final (most recent) row.
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=120,
                                              learning_rate=0.08, random_state=42)
        clf.fit(X[:-1], y[:-1])
        proba_up = float(clf.predict_proba(X[-1:])[0][list(clf.classes_).index(1)])
    except Exception as exc:
        logger.debug("FiscalAI: model training failed (%s); using heuristic.", exc)
        return None

    # Map probability-up in [0,1] -> signal in [-1,1]; confidence = distance from 0.5.
    signal = (proba_up - 0.5) * 2.0
    confidence = abs(proba_up - 0.5) * 2.0
    return {"signal": signal, "confidence": confidence, "proba_up": proba_up}


def _heuristic_signal(feat: pd.DataFrame, trend: Dict[str, float]) -> Dict[str, float]:
    """Momentum/trend blend used when the model can't train."""
    last = feat.dropna().iloc[-1] if not feat.dropna().empty else None
    components = []
    if last is not None:
        components.append(np.tanh(last.get("ma_ratio_50", 0.0) * 10))
        rsi = last.get("rsi", 50.0)
        components.append((rsi - 50.0) / 50.0)
        components.append(np.tanh(last.get("macd_hist", 0.0) * 50))
    components.append(np.tanh(trend["slope"] * 50))
    signal = float(np.clip(np.mean(components) if components else 0.0, -1.0, 1.0))
    confidence = min(1.0, 0.3 + trend["r2"] * 0.5)
    return {"signal": signal, "confidence": confidence}


class FiscalAI:
    """Quantitative engine: statistics, risk, and a directional signal."""

    name = "Fiscal Quantitative Engine"

    def analyze(self, symbol: str) -> Optional[FiscalInsight]:
        df = fetch_ohlcv(symbol, period="2y", interval="1d")
        if df is None or "Close" not in df.columns:
            logger.warning("FiscalAI: no usable data for %s.", symbol)
            return None
        return self.score_dataframe(symbol, df)

    def score_dataframe(self, symbol: str, df: pd.DataFrame) -> FiscalInsight:
        close = df["Close"].astype(float).dropna()
        feat = _build_features(df)
        risk = _risk_metrics(close)
        trend = _trend(close)

        model_out = _model_signal(feat, close)
        if model_out is not None:
            signal, confidence, model_name = model_out["signal"], model_out["confidence"], "HistGradientBoosting"
            proba_up = model_out["proba_up"]
        else:
            h = _heuristic_signal(feat, trend)
            signal, confidence, model_name = h["signal"], h["confidence"], "heuristic"
            proba_up = float(np.clip((signal + 1.0) / 2.0, 0.0, 1.0))  # derive from signal

        mas = {}
        for w in (20, 50, 200):
            if len(close) >= w:
                mas[f"MA{w}"] = round(float(close.rolling(w).mean().iloc[-1]), 4)

        # Latest (most recent) feature row, NaNs coerced to 0.
        feat_keys = ["rsi", "macd_hist", "bb_pos", "volume_z", "ma_ratio_20", "ma_ratio_50", "ma_ratio_200"]
        features: Dict[str, float] = {k: 0.0 for k in feat_keys}
        if not feat.empty:
            last_row = feat.iloc[-1]
            for k in feat_keys:
                if k in feat.columns:
                    val = last_row.get(k)
                    features[k] = float(val) if val is not None and not np.isnan(val) else 0.0

        # Market context.
        volume = float(df["Volume"].dropna().iloc[-1]) if "Volume" in df.columns and not df["Volume"].dropna().empty else 0.0
        day_high = float(df["High"].dropna().iloc[-1]) if "High" in df.columns and not df["High"].dropna().empty else 0.0
        day_low = float(df["Low"].dropna().iloc[-1]) if "Low" in df.columns and not df["Low"].dropna().empty else 0.0
        window52 = close.tail(252)
        week52_high = float(window52.max()) if not window52.empty else 0.0
        week52_low = float(window52.min()) if not window52.empty else 0.0

        return FiscalInsight(
            symbol=symbol.upper(),
            signal=round(float(np.clip(signal, -1.0, 1.0)), 4),
            confidence=round(float(np.clip(confidence, 0.0, 1.0)), 4),
            last_price=round(float(close.iloc[-1]), 4),
            sharpe_ratio=round(risk["sharpe"], 4),
            var_95=round(risk["var_95"], 6),
            trend_slope=round(trend["slope"], 8),
            trend_r2=round(trend["r2"], 4),
            mean_return=round(risk["mean_return"], 6),
            volatility=round(risk["volatility"], 6),
            moving_averages=mas,
            model=model_name,
            features={k: round(v, 6) for k, v in features.items()},
            proba_up=round(float(np.clip(proba_up, 0.0, 1.0)), 4),
            volume=volume,
            day_high=round(day_high, 4),
            day_low=round(day_low, 4),
            week52_high=round(week52_high, 4),
            week52_low=round(week52_low, 4),
        )
