"""Config tests."""
from proverbs.config import Settings


def test_normalised_weights_sum_to_one():
    s = Settings(fiscal_weight=0.65, cultural_weight=0.35)
    fw, cw = s.normalised_weights
    assert round(fw + cw, 6) == 1.0

    s2 = Settings(fiscal_weight=3.0, cultural_weight=1.0)
    fw2, cw2 = s2.normalised_weights
    assert round(fw2 + cw2, 6) == 1.0
    assert fw2 > cw2


def test_postgres_url_normalised():
    s = Settings()
    from proverbs.config import _normalise_db_url
    assert _normalise_db_url("postgres://u:p@host/db").startswith("postgresql://")


def test_watchlist_default_present():
    s = Settings()
    assert len(s.watchlist) >= 1
