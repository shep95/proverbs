"""News / headline feed.

Phase 1 of the rebuild: the original scraped ``h3`` tags off JS-rendered finance
homepages (unreliable, frequently empty). This replaces that with structured RSS
feeds (keyless, stable) as the default, plus an optional NewsAPI backend.

Returns a list of ``Headline`` objects the Cultural AI turns into a sentiment index.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from proverbs.config import settings

logger = logging.getLogger(__name__)

# Keyless RSS sources. Generic market feeds plus a per-symbol Yahoo feed.
_MARKET_RSS = [
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",          # WSJ Markets
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",  # CNBC Top News
    "https://feeds.marketwatch.com/marketwatch/topstories/",  # MarketWatch
]


def _yahoo_symbol_feed(symbol: str) -> str:
    return f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"


@dataclass
class Headline:
    title: str
    source: str
    published: Optional[datetime] = None
    symbol: Optional[str] = None


def _parse_time(entry) -> Optional[datetime]:
    for attr in ("published_parsed", "updated_parsed"):
        tp = getattr(entry, attr, None)
        if tp:
            try:
                return datetime(*tp[:6], tzinfo=timezone.utc)
            except Exception:
                continue
    return None


def _fetch_rss(url: str, symbol: Optional[str], limit: int) -> List[Headline]:
    try:
        import feedparser

        parsed = feedparser.parse(url)
    except Exception as exc:
        logger.warning("news_feed: failed to parse %s: %s", url, exc)
        return []

    source = getattr(parsed.feed, "title", url) if getattr(parsed, "feed", None) else url
    out: List[Headline] = []
    for entry in parsed.entries[:limit]:
        title = (getattr(entry, "title", "") or "").strip()
        if title:
            out.append(Headline(title=title, source=source, published=_parse_time(entry), symbol=symbol))
    return out


def _fetch_newsapi(symbol: Optional[str], limit: int) -> List[Headline]:
    if not settings.newsapi_key:
        logger.warning("news_feed: NEWS_BACKEND=newsapi but no NEWSAPI_KEY; skipping.")
        return []
    try:
        import requests

        params = {
            "apiKey": settings.newsapi_key,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": limit,
            "q": symbol or "stock market",
        }
        resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=15)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
    except Exception as exc:
        logger.warning("news_feed: NewsAPI request failed: %s", exc)
        return []

    out: List[Headline] = []
    for art in articles[:limit]:
        title = (art.get("title") or "").strip()
        if title:
            out.append(
                Headline(
                    title=title,
                    source=(art.get("source") or {}).get("name", "NewsAPI"),
                    symbol=symbol,
                )
            )
    return out


def fetch_headlines(symbol: Optional[str] = None, limit: int = 20) -> List[Headline]:
    """Fetch recent headlines, optionally scoped to ``symbol``.

    Backend selected by ``NEWS_BACKEND`` (rss default, newsapi optional).
    Always returns a list (possibly empty) — callers degrade gracefully.
    """
    if settings.news_backend == "newsapi":
        headlines = _fetch_newsapi(symbol, limit)
        if headlines:
            return headlines
        logger.info("news_feed: NewsAPI empty, falling back to RSS.")

    headlines: List[Headline] = []
    if symbol:
        headlines.extend(_fetch_rss(_yahoo_symbol_feed(symbol), symbol, limit))
    # Top up with general market feeds so an illiquid ticker still gets signal.
    remaining = max(0, limit - len(headlines))
    per_feed = max(3, remaining // max(1, len(_MARKET_RSS)))
    for url in _MARKET_RSS:
        if len(headlines) >= limit:
            break
        headlines.extend(_fetch_rss(url, symbol, per_feed))

    # De-duplicate on title, preserve order.
    seen = set()
    unique: List[Headline] = []
    for h in headlines:
        key = h.title.lower()
        if key not in seen:
            seen.add(key)
            unique.append(h)
    return unique[:limit]
