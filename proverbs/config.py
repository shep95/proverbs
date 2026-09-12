"""Central configuration, loaded once from environment variables.

Import ``settings`` anywhere:  ``from proverbs.config import settings``.
No secrets are ever hardcoded here — the only required one is DISCORD_TOKEN.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # python-dotenv is optional at runtime
    pass


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _env_int(key: str, default: Optional[int]) -> Optional[int]:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_list(key: str, default: List[str]) -> List[str]:
    raw = os.getenv(key)
    if not raw:
        return default
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def _normalise_db_url(url: str) -> str:
    # Railway/Heroku hand out ``postgres://`` which SQLAlchemy no longer accepts.
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url


@dataclass
class Settings:
    # --- Discord ---
    discord_token: str = field(default_factory=lambda: os.getenv("DISCORD_TOKEN", ""))
    discord_guild_id: Optional[int] = field(default_factory=lambda: _env_int("DISCORD_GUILD_ID", None))
    alert_channel_id: Optional[int] = field(default_factory=lambda: _env_int("DISCORD_ALERT_CHANNEL_ID", None))
    command_prefix: str = field(default_factory=lambda: os.getenv("DISCORD_COMMAND_PREFIX", "!"))

    # --- Engine ---
    watchlist: List[str] = field(default_factory=lambda: _env_list("WATCHLIST", ["AAPL", "MSFT", "GOOGL", "TSLA"]))
    cycle_interval_minutes: int = field(default_factory=lambda: int(_env_float("CYCLE_INTERVAL_MINUTES", 30)))
    fiscal_weight: float = field(default_factory=lambda: _env_float("FISCAL_WEIGHT", 0.65))
    cultural_weight: float = field(default_factory=lambda: _env_float("CULTURAL_WEIGHT", 0.35))
    buy_threshold: float = field(default_factory=lambda: _env_float("BUY_THRESHOLD", 0.3))
    reduce_threshold: float = field(default_factory=lambda: _env_float("REDUCE_THRESHOLD", -0.3))

    # --- Auto-withdrawal (paper simulation) ---
    auto_withdraw_multiple: float = field(default_factory=lambda: _env_float("AUTO_WITHDRAW_MULTIPLE", 5.5))
    auto_withdraw_fraction: float = field(default_factory=lambda: _env_float("AUTO_WITHDRAW_FRACTION", 0.2))

    # --- Persistence ---
    database_url: str = field(
        default_factory=lambda: _normalise_db_url(os.getenv("DATABASE_URL", "sqlite:///proverbs.db"))
    )

    # --- Backends ---
    cultural_backend: str = field(default_factory=lambda: os.getenv("CULTURAL_BACKEND", "vader").lower())
    news_backend: str = field(default_factory=lambda: os.getenv("NEWS_BACKEND", "rss").lower())
    newsapi_key: str = field(default_factory=lambda: os.getenv("NEWSAPI_KEY", ""))

    # --- Web ---
    port: int = field(default_factory=lambda: int(_env_float("PORT", 8080)))
    enable_web: bool = field(default_factory=lambda: _env_bool("ENABLE_WEB", True))

    # --- Logging ---
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper())

    @property
    def normalised_weights(self) -> tuple[float, float]:
        """Return (fiscal, cultural) weights re-normalised to sum to 1.0."""
        total = self.fiscal_weight + self.cultural_weight
        if total <= 0:
            return 0.65, 0.35
        return self.fiscal_weight / total, self.cultural_weight / total

    def validate(self) -> list[str]:
        """Return a list of human-readable configuration problems (empty == OK)."""
        problems: list[str] = []
        if not self.discord_token:
            problems.append("DISCORD_TOKEN is not set — the Discord bot cannot start.")
        if self.cultural_backend not in {"vader", "transformers"}:
            problems.append(f"CULTURAL_BACKEND '{self.cultural_backend}' is invalid (use vader|transformers).")
        if self.news_backend not in {"rss", "newsapi"}:
            problems.append(f"NEWS_BACKEND '{self.news_backend}' is invalid (use rss|newsapi).")
        if self.news_backend == "newsapi" and not self.newsapi_key:
            problems.append("NEWS_BACKEND=newsapi but NEWSAPI_KEY is empty.")
        if not self.watchlist:
            problems.append("WATCHLIST is empty.")
        return problems


settings = Settings()


def configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # yfinance / urllib3 are chatty; keep them at WARNING.
    for noisy in ("urllib3", "yfinance", "peewee", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
