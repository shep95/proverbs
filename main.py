"""proverbs entrypoint.

Runs the Flask dashboard (via waitress) in a daemon thread and the Discord bot
on the main thread's asyncio loop, so a single Railway process serves both.

  * DISCORD_TOKEN set              -> bot + (optional) dashboard
  * DISCORD_TOKEN missing          -> dashboard only, with a clear warning
  * ENABLE_WEB=false               -> bot only
"""
from __future__ import annotations

import asyncio
import logging
import threading

from proverbs.config import configure_logging, settings
from proverbs.persistence import db

logger = logging.getLogger("proverbs")


def _start_web_thread() -> None:
    from waitress import serve

    from proverbs.web.api import create_app

    app = create_app()

    def _run():
        logger.info("Dashboard listening on 0.0.0.0:%s", settings.port)
        serve(app, host="0.0.0.0", port=settings.port, threads=8, _quiet=True)

    t = threading.Thread(target=_run, name="web", daemon=True)
    t.start()


def main() -> None:
    configure_logging()
    logger.info("Starting proverbs v%s", __import__("proverbs").__version__)

    problems = settings.validate()
    for p in problems:
        logger.warning("config: %s", p)

    db.init_db()

    # Connect the broker (paper by default; Robinhood if configured). Safe to call
    # even without credentials — it degrades to trading-disabled.
    from proverbs.trading.manager import trading

    trading.init()
    logger.info("Broker=%s · mode=%s · auto_trade=%s",
                settings.broker, "LIVE" if trading.state.live else "DRY-RUN", settings.auto_trade)

    # If a paper-trading session was running before a restart, resume it.
    from proverbs.trading.session import sessions

    sessions.resume_active()

    if settings.enable_web:
        _start_web_thread()

    if not settings.discord_token:
        if not settings.enable_web:
            raise SystemExit("DISCORD_TOKEN is not set and ENABLE_WEB=false — nothing to run.")
        logger.warning("DISCORD_TOKEN not set — running dashboard only. Set it to enable Discord.")
        # Keep the process alive so the web thread keeps serving.
        threading.Event().wait()
        return

    from proverbs.bot.discord_bot import start_bot

    try:
        asyncio.run(start_bot())
    except KeyboardInterrupt:
        logger.info("Shutting down.")


if __name__ == "__main__":
    main()
