"""Test configuration — force a throwaway SQLite DB before proverbs imports."""
import os
import tempfile

# Must be set BEFORE any proverbs module imports (config reads env at import time).
_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp.close()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp.name}")
os.environ.setdefault("DISCORD_TOKEN", "test-token")
os.environ.setdefault("CULTURAL_BACKEND", "vader")
os.environ.setdefault("NEWS_BACKEND", "rss")
