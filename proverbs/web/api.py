"""Flask API + dashboard.

Phase 6-7 of the rebuild: every route returns real data from the engine/DB
(no more ``random.choice``), with validation and structured JSON. The dashboard
(templates/index.html) polls these endpoints for a live view.

The dashboard drives the shared "global" account so anyone viewing the web UI
sees the same simulated portfolio the engine grows each cycle. Per-user paper
accounts live in Discord.
"""
from __future__ import annotations

import logging

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from proverbs.config import settings
from proverbs.persistence import db
from proverbs.services.engine import engine

logger = logging.getLogger(__name__)

DASHBOARD_ACCOUNT = db.GLOBAL_ACCOUNT_ID


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    CORS(app)

    @app.route("/")
    def index():
        return render_template("index.html", watchlist=settings.watchlist)

    @app.route("/healthz")
    def healthz():
        return jsonify({"status": "ok", "service": "proverbs"})

    @app.route("/api/signal")
    def api_signal():
        symbol = request.args.get("symbol")
        if symbol:
            return jsonify(engine.signal_history(symbol=symbol, limit=30))
        return jsonify(engine.latest_signals())

    @app.route("/api/balance")
    def api_balance():
        return jsonify(engine.get_account(DASHBOARD_ACCOUNT, "Global"))

    @app.route("/api/portfolio")
    def api_portfolio():
        return jsonify(engine.get_portfolio(DASHBOARD_ACCOUNT))

    @app.route("/api/history")
    def api_history():
        return jsonify(engine.get_history(DASHBOARD_ACCOUNT, limit=25))

    @app.route("/api/deposit", methods=["POST"])
    def api_deposit():
        amount = _amount_from_request()
        if amount is None:
            return jsonify({"status": "error", "message": "Invalid amount"}), 400
        res = engine.deposit(DASHBOARD_ACCOUNT, amount, "Global")
        return jsonify(res), (200 if res["status"] == "success" else 400)

    @app.route("/api/withdraw", methods=["POST"])
    def api_withdraw():
        amount = _amount_from_request()
        if amount is None:
            return jsonify({"status": "error", "message": "Invalid amount"}), 400
        res = engine.withdraw(DASHBOARD_ACCOUNT, amount, "Global")
        return jsonify(res), (200 if res["status"] == "success" else 400)

    @app.route("/api/risk", methods=["POST"])
    def api_risk():
        data = request.get_json(silent=True) or request.form
        level = (data.get("risk_level") or "").strip()
        res = engine.set_risk(DASHBOARD_ACCOUNT, level, "Global")
        return jsonify(res), (200 if res["status"] == "success" else 400)

    @app.route("/api/trading")
    def api_trading():
        from proverbs.trading.manager import trading
        return jsonify(trading.status())

    @app.route("/api/positions")
    def api_positions():
        from proverbs.trading.manager import trading
        if trading.broker is None:
            return jsonify([])
        return jsonify([p.to_dict() for p in trading.broker.get_positions()])

    @app.route("/api/cycle", methods=["POST"])
    def api_cycle():
        """Trigger an analysis cycle on demand (useful for testing/demo)."""
        report = engine.run_cycle()
        return jsonify({
            "status": "success",
            "avg_score": report.avg_score,
            "simulated_return_pct": report.simulated_return_pct,
            "balance_after": report.balance_after,
            "decisions": [
                {"symbol": d.symbol, "action": d.action, "combined_score": d.combined_score,
                 "confidence": d.confidence}
                for d in report.decisions
            ],
            "errors": report.errors,
        })

    return app


def _amount_from_request():
    data = request.get_json(silent=True) or request.form
    raw = data.get("amount")
    try:
        amount = float(raw)
    except (TypeError, ValueError):
        return None
    return amount
