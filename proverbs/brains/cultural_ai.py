"""Cultural AI — the sentiment / social-intelligence brain.

Phase 2 of the rebuild: score each headline, bucket into pos/neutral/neg, and
compute a single weighted **cultural index in [-1.0, 1.0]** with recent
headlines weighted more heavily via exponential decay. This float feeds the
orchestrator — it is no longer just printed to the console.

Two backends:
  * ``vader`` (default) — VADER lexicon sentiment; tiny, fast, no torch.
  * ``transformers`` — Hugging Face ``sentiment-analysis`` pipeline (opt-in;
    requires torch, see requirements.txt). Loaded lazily and cached.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional

from proverbs.config import settings
from proverbs.feeds.news_feed import Headline, fetch_headlines

logger = logging.getLogger(__name__)


@dataclass
class CulturalInsight:
    symbol: Optional[str]
    index: float                 # weighted sentiment in [-1, 1]
    positive: int
    neutral: int
    negative: int
    sample_size: int
    headlines: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        if self.index > 0.15:
            return "Bullish"
        if self.index < -0.15:
            return "Bearish"
        return "Neutral"


class _VaderBackend:
    def __init__(self) -> None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self._analyzer = SentimentIntensityAnalyzer()

    def score(self, text: str) -> float:
        # compound is already in [-1, 1].
        return float(self._analyzer.polarity_scores(text)["compound"])


class _TransformersBackend:
    def __init__(self) -> None:
        from transformers import pipeline

        self._pipe = pipeline("sentiment-analysis")

    def score(self, text: str) -> float:
        result = self._pipe(text[:512])[0]
        label = result["label"].upper()
        conf = float(result["score"])
        if label.startswith("POS"):
            return conf
        if label.startswith("NEG"):
            return -conf
        return 0.0


class CulturalAI:
    """Reads market culture and emits a numeric sentiment signal."""

    name = "Cultural Insight Engine"

    def __init__(self) -> None:
        self._backend = None  # lazy

    def _get_backend(self):
        if self._backend is not None:
            return self._backend
        try:
            if settings.cultural_backend == "transformers":
                self._backend = _TransformersBackend()
                logger.info("CulturalAI: using Hugging Face transformers backend.")
            else:
                self._backend = _VaderBackend()
                logger.info("CulturalAI: using VADER backend.")
        except Exception as exc:
            logger.warning("CulturalAI: backend '%s' unavailable (%s); falling back to VADER.",
                           settings.cultural_backend, exc)
            self._backend = _VaderBackend()
        return self._backend

    def analyze(self, symbol: Optional[str] = None, limit: int = 20) -> CulturalInsight:
        """Fetch headlines for ``symbol`` and return a weighted cultural insight."""
        headlines = fetch_headlines(symbol=symbol, limit=limit)
        return self.score_headlines(headlines, symbol=symbol)

    def score_headlines(self, headlines: List[Headline], symbol: Optional[str] = None) -> CulturalInsight:
        if not headlines:
            logger.info("CulturalAI: no headlines for %s; neutral index.", symbol or "market")
            return CulturalInsight(symbol=symbol, index=0.0, positive=0, neutral=0,
                                   negative=0, sample_size=0)

        backend = self._get_backend()
        scores: List[float] = []
        pos = neu = neg = 0
        for h in headlines:
            try:
                s = backend.score(h.title)
            except Exception as exc:
                logger.debug("CulturalAI: scoring failed for a headline: %s", exc)
                s = 0.0
            scores.append(s)
            if s > 0.05:
                pos += 1
            elif s < -0.05:
                neg += 1
            else:
                neu += 1

        # Exponential-decay weighting: newest headline (index 0) weighted highest.
        # Assumes fetch_headlines returns newest-first (RSS/NewsAPI both do).
        decay = 0.9
        weights = [decay ** i for i in range(len(scores))]
        wsum = sum(weights) or 1.0
        index = sum(s * w for s, w in zip(scores, weights)) / wsum
        index = max(-1.0, min(1.0, index))

        return CulturalInsight(
            symbol=symbol,
            index=round(index, 4),
            positive=pos,
            neutral=neu,
            negative=neg,
            sample_size=len(scores),
            headlines=[h.title for h in headlines[:5]],
        )
