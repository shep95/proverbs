"""Market data feed.

Phase 1 of the rebuild: replace the deprecated Yahoo CSV download endpoint
(``query1.finance.yahoo.com/v7/finance/download/...`` — now 401/403s) with the
maintained ``yfinance`` library, which handles splits/dividends and returns a
clean OHLCV DataFrame. Results are cached briefly to avoid hammering the API
when several brains/commands ask for the same ticker in one cycle.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ticker -> (fetched_at_epoch, dataframe)
_CACHE: Dict[Tuple[str, str], Tuple[float, pd.DataFrame]] = {}
_CACHE_TTL_SECONDS = 60 * 10  # 10 minutes


@dataclass
class PriceSnapshot:
    symbol: str
    last_price: float
    prev_close: float

    @property
    def change_pct(self) -> float:
        if self.prev_close == 0:
            return 0.0
        return (self.last_price - self.prev_close) / self.prev_close * 100.0


def fetch_ohlcv(symbol: str, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
    """Return an OHLCV DataFrame for ``symbol`` (or ``None`` on failure).

    Columns are normalised to: Open, High, Low, Close, Volume, with a DatetimeIndex.
    """
    symbol = symbol.upper().strip()
    key = (symbol, f"{period}:{interval}")
    now = time.time()
    cached = _CACHE.get(key)
    if cached and now - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]

    try:
        import yfinance as yf  # imported lazily so the package imports without it

        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as exc:  # network, rate-limit, bad symbol, etc.
        logger.warning("market_feed: failed to fetch %s: %s", symbol, exc)
        return None

    if df is None or df.empty:
        logger.warning("market_feed: no data returned for %s", symbol)
        return None

    # yfinance may return a MultiIndex column set when given a single symbol.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.title)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].dropna(how="all")

    _CACHE[key] = (now, df)
    return df


def latest_snapshot(symbol: str) -> Optional[PriceSnapshot]:
    """Return the most recent price + previous close for ``symbol``."""
    df = fetch_ohlcv(symbol, period="5d", interval="1d")
    if df is None or "Close" in df.columns and len(df["Close"].dropna()) < 1:
        return None
    closes = df["Close"].dropna()
    if closes.empty:
        return None
    last_price = float(closes.iloc[-1])
    prev_close = float(closes.iloc[-2]) if len(closes) >= 2 else last_price
    return PriceSnapshot(symbol=symbol.upper(), last_price=last_price, prev_close=prev_close)


def clear_cache() -> None:
    _CACHE.clear()
