"""Trading layer: broker abstraction, paper + Robinhood adapters, risk manager.

Signals from the orchestrator are turned into orders here, behind a strict set
of safeguards. Defaults are safe: BROKER=paper and LIVE_TRADING=false (dry-run).
"""
