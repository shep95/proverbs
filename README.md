# 📜 proverbs

A three-brain algorithmic **investment signal bot** for Discord, hosted on Railway.

`proverbs` reads market *culture* (news sentiment), scores the *quantitative*
risk/reward of each ticker, and an *orchestrator* fuses the two into a
**BUY / HOLD / REDUCE** call. It posts scheduled alerts to a Discord channel and
answers slash commands, and it ships a live web dashboard as a secondary view.

> ⚠️ **Not financial advice.** `proverbs` ships **safe by default** — it paper-trades
> and runs in dry-run, placing **no real orders** until you explicitly set
> `BROKER=robinhood` and `LIVE_TRADING=true`. Live trading uses the *unofficial*
> `robin_stocks` client (against Robinhood's ToS) and risks **real losses** — see
> [Real trading](#real-trading-robinhood-equities). Use at your own risk.

---

## Architecture

```
        news RSS / NewsAPI              yfinance OHLCV
               │                              │
        ┌──────▼───────┐              ┌───────▼────────┐
        │  Cultural AI │              │   Fiscal AI    │
        │  sentiment   │              │  features +    │
        │  index -1..1 │              │  Sharpe + VaR  │
        └──────┬───────┘              │  + ML signal   │
               │                      └───────┬────────┘
               └───────────┬──────────────────┘
                           ▼
                   ┌───────────────┐
                   │ Orchestrator  │  weighted score → BUY/HOLD/REDUCE
                   │ + auto-withdraw│  + confidence
                   └───────┬───────┘
                           ▼
                  ┌──────────────────┐
                  │  Persistence (DB)│  accounts · positions · signals · txns
                  └───────┬──────────┘
              ┌───────────┴────────────┐
              ▼                         ▼
       Discord bot                 Flask dashboard
   (alerts + commands)            (chart.js, live poll)
```

### The three brains
- **Cultural AI** (`proverbs/brains/cultural_ai.py`) — scores recent headlines and
  produces a single sentiment index in `[-1, 1]`, weighting newer headlines more
  (exponential decay). Backend is **VADER** by default (tiny, fast); set
  `CULTURAL_BACKEND=transformers` to use a Hugging Face pipeline instead.
- **Fiscal AI** (`proverbs/brains/fiscal_ai.py`) — engineers a feature set
  (20/50/200-day MA ratios, RSI, MACD, volume z-score, Bollinger position, returns,
  volatility) and trains a scikit-learn `HistGradientBoosting` classifier to predict
  next-day direction, yielding a signal + confidence. Also computes the
  **Sharpe ratio** and **95% VaR** (kept from the original Edward-Thorp engine).
  Falls back to a trend/momentum heuristic when history is short.
- **Orchestrator** (`proverbs/brains/orchestrator.py`) — `combined = 0.65·fiscal +
  0.35·cultural` (weights configurable), thresholded into BUY/HOLD/REDUCE, and owns
  the auto-withdrawal rule (withdraw 20% of profit once balance ≥ 5.5× deposit).

---

## Discord commands

| Command | What it does |
| --- | --- |
| `/signal [ticker]` | Latest signal(s); pass a ticker for a fresh single-symbol read |
| `/analyze <ticker>` | Run a full analysis on a ticker right now |
| `/runcycle` | Run a full analyze-and-trade cycle on demand |
| `/top` | Rank the whole watchlist by combined score |
| `/compare <a> <b>` | Side-by-side signal comparison of two tickers |
| `/signalhistory <ticker> [limit]` | How a ticker's signal evolved, with ✅/❌ accuracy marks |
| `/explain <ticker>` | Plain-language breakdown of *why* the bot made the call |
| `/balance` `/deposit` `/withdraw` `/risk` `/portfolio` `/history` | Paper account |
| `/watchlist show\|add\|remove` | View or edit the tracked tickers at runtime |
| `/setweights <fiscal> <cultural>` | Adjust the fiscal/cultural blend at runtime |
| `/alert <ticker> <above\|below> <score>` · `/alerts` | Per-user score alerts |
| `/help` · `/ping` | Help / liveness |

Scheduled alerts (notable BUY/REDUCE calls + auto-withdrawals + orders) post
automatically to `DISCORD_ALERT_CHANNEL_ID` every `CYCLE_INTERVAL_MINUTES`.

### Trading commands
| Command | What it does |
| --- | --- |
| `/trading` | Broker status, mode (LIVE/DRY-RUN), account, and risk limits |
| `/positions` | Current broker positions with P/L |
| `/order <buy\|sell> <ticker> <amount>` | Place a manual order (respects all safeguards) |
| `/mode <dry-run\|live>` | Switch between dry-run and LIVE trading |
| `/kill` · `/resume` | Halt / resume all trading instantly (runtime kill switch) |
| `/papertrade start\|status\|stop\|report\|leaderboard` | Investor paper-trading sessions |

---

## Analytics & transparency

Every signal is now stored with its **full context**, so calls can be audited and
the model graded — not just `BUY`/`HOLD`/`REDUCE`:

- **Feature vector:** RSI, MACD histogram, Bollinger position, volume z-score,
  20/50/200-day MA ratios, and the raw model probability `P(up)`.
- **Market context:** volume, day high/low, 52-week high/low.
- **Cultural detail:** positive/neutral/negative counts, sample size, news
  sources, and the top headlines behind the sentiment index.

**Signal-accuracy tracking.** The model predicts next-day direction; each cycle
the engine looks back and grades earlier signals against the realized price once
`ACCURACY_HORIZON_HOURS` (default 24) have passed — recording whether the call
was right. Hit-rate shows up in `/signalhistory`, `/explain`, the investor report,
and `GET /api/accuracy`. The horizon is per-symbol overridable via
`ACCURACY_HORIZON_OVERRIDES="TSLA=6,AAPL=48"` (hours).

**Richer investor report.** `/papertrade report` and `/report` now include
**win rate, best/worst trade, average hold time** (from a realized-trade ledger)
and **signal accuracy**, alongside return, drawdown, Sharpe, and the equity curve.
`/papertrade leaderboard [by:session|user]` ranks finished sessions — or users
across all their sessions (avg + best return) — by return %.

---

## Real trading (Robinhood, equities)

`proverbs` can route its BUY/HOLD/REDUCE decisions to a broker. **Everything is
safe by default** — `BROKER=paper` and `LIVE_TRADING=false` (dry-run) — so no real
money moves until you deliberately enable it.

```
decision → size order → risk checks → LIVE gate → broker
  (BUY/REDUCE)   (confidence)  (limits, hours,   (dry-run logs;    (paper | robinhood)
                                kill switch)      live sends)
```

### ⚠️ Read this before going live
- Robinhood has **no official stock API**. This uses **`robin_stocks`**, an
  *unofficial, reverse-engineered* client — it is **against Robinhood's ToS**, can
  break without notice, and can get an account flagged. **Use at your own risk.**
- Real orders mean **real losses**. `proverbs` is provided as-is, **not financial
  advice**, with no warranty. Start in dry-run, then paper, then tiny live sizes.

### What you need for the Robinhood system
1. A **funded Robinhood account**.
2. **App-based 2FA**: in Robinhood, enable Two-Factor → *Authenticator App*, and
   copy the **setup/secret key** — headless login on Railway can't do SMS/tap 2FA,
   so it generates TOTP codes from this secret (`ROBINHOOD_MFA_SECRET`).
3. Set these Railway Variables (never commit them):
   - `BROKER=robinhood`
   - `ROBINHOOD_USERNAME`, `ROBINHOOD_PASSWORD`, `ROBINHOOD_MFA_SECRET`
   - `LIVE_TRADING=false` at first (dry-run) — flip to `true` only when ready
4. Tune the **risk limits** (all in `.env.example`): `BASE_ORDER_NOTIONAL`,
   `MAX_ORDER_NOTIONAL`, `MAX_POSITION_NOTIONAL`, `MAX_OPEN_POSITIONS`,
   `DAILY_LOSS_LIMIT`, `ORDER_CONFIDENCE_MIN`.

### Built-in safeguards
- **Dry-run by default** — intended orders are logged, never sent, until `LIVE_TRADING=true`.
- **Kill switch** (`/kill`, `TRADING_ENABLED=false`) halts all orders instantly.
- **Per-order & per-position caps**, **max open positions**, **min confidence**.
- **Daily loss limit** trips the kill switch automatically.
- **US market-hours guard** (holidays not tracked — set `IGNORE_MARKET_HOURS=true` to bypass).
- Order sizing scales with signal confidence, capped by your limits.

### Test it completely first — live paper-trading (no real money)
The paper broker runs the **entire pipeline** — real signals, real market prices,
scheduled cycles, orders, positions, P/L — against a simulated wallet. Nothing
touches a real account. (Paper trades any time, including nights/weekends, using
the last close; the market-hours guard only applies to real brokers.)

**The easiest way: a `/papertrade` session.** In Discord:
```
/papertrade start amount:10000 duration:7d      # start a tracked run
/papertrade status                              # live performance any time
/papertrade report                              # full report + downloadable JSON
/papertrade stop                                # end early
```
A session resets a fresh $10k (your `amount`) paper wallet, turns on live
paper-trading, trades every cycle for the `duration` (e.g. `90m`, `24h`, `7d`),
snapshots the equity curve, and auto-stops when time's up. It survives Railway
restarts (it resumes automatically).

**Shareholder report.** Every session produces an investor-ready report:
starting capital → current equity, return %, net P/L, max drawdown, Sharpe,
trade count, open positions, and an equity-curve chart. Share it two ways:
- **Live web page:** `https://<your-railway-domain>/report` (auto-refreshes) — send
  this link to investors.
- **In Discord:** `/papertrade report` posts the summary and attaches the full
  `*_report.json` data file.

Prefer manual control instead of a timed session? Set `BROKER=paper`,
`LIVE_TRADING=true`, `AUTO_TRADE=true` and use **`/runcycle`** to trade on demand,
watching `/positions`, `/trading`, `/history` and the dashboard.

When you're happy it behaves, switch `BROKER=robinhood` (still start in dry-run).

### Recommended rollout
1. `BROKER=paper`, dry-run → watch the signals and intended orders in alerts.
2. `BROKER=paper`, `LIVE_TRADING=true` → exercise the full order path on the
   simulated account (`PAPER_STARTING_CASH`).
3. `BROKER=robinhood`, dry-run → confirm login + prices + positions read correctly.
4. `BROKER=robinhood`, `LIVE_TRADING=true`, **small** `MAX_ORDER_NOTIONAL` → go live.

> **A note on brokers:** Robinhood's stock API is unofficial and risky. If you
> ever want a supported path, **Alpaca** offers official API keys plus a true
> paper-trading sandbox and slots into the same `Broker` interface
> (`proverbs/trading/base.py`) — just add an adapter.

---

## Setup

### 1. Create the Discord bot
1. Go to <https://discord.com/developers/applications> → **New Application**.
2. **Bot** tab → **Add Bot** → copy the **token** (this is your `DISCORD_TOKEN`).
3. Under **Privileged Gateway Intents**, enable **Message Content Intent**
   (needed for classic prefix commands; slash commands work regardless).
4. **OAuth2 → URL Generator**: scopes `bot` + `applications.commands`;
   bot permissions `Send Messages`, `Embed Links`. Use the generated URL to invite
   the bot to your server.
5. To get a channel ID for alerts, enable **Developer Mode** (User Settings →
   Advanced), right-click the channel → **Copy ID**.

### 2. Deploy on Railway
1. Push this repo to GitHub and create a **New Project → Deploy from GitHub repo**
   on <https://railway.app>.
2. (Recommended) Add a **PostgreSQL** plugin — Railway injects `DATABASE_URL`
   automatically, so your data survives restarts and redeploys. Without it,
   `proverbs` uses a local SQLite file (fine for testing, wiped on redeploy).
3. Under **Variables**, set at minimum:
   - `DISCORD_TOKEN` — **required**
   - `DISCORD_ALERT_CHANNEL_ID` — channel for scheduled alerts (optional)
   - `DISCORD_GUILD_ID` — your server ID for instant slash-command sync (optional)
   - Any tuning from [`.env.example`](.env.example) (watchlist, weights, etc.)
4. Railway builds via Nixpacks and runs `python main.py` (see `railway.json` /
   `Procfile`). The web dashboard is exposed on Railway's generated URL.

> **On API keys:** the only secret you must provide is `DISCORD_TOKEN`.
> Market data (`yfinance`) and news (`RSS`) need no keys. NewsAPI and the Hugging
> Face backend are optional. **Secrets are never committed** — set them as Railway
> Variables (or a local `.env`, which is git-ignored). Do not hardcode tokens in
> source; anything pushed to git is effectively public forever.

### 3. Local development
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then fill in DISCORD_TOKEN
python main.py                # bot + dashboard at http://localhost:8080
```
Run the bot without the dashboard with `ENABLE_WEB=false`, or the dashboard
without the bot by leaving `DISCORD_TOKEN` empty.

---

## Configuration

All settings come from environment variables — see [`.env.example`](.env.example)
for the full annotated list. Highlights:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DISCORD_TOKEN` | — | **Required.** Bot token |
| `WATCHLIST` | `AAPL,MSFT,GOOGL,TSLA` | Tickers analysed each cycle |
| `CYCLE_INTERVAL_MINUTES` | `30` | How often the engine runs |
| `FISCAL_WEIGHT` / `CULTURAL_WEIGHT` | `0.65` / `0.35` | Orchestrator blend |
| `BUY_THRESHOLD` / `REDUCE_THRESHOLD` | `0.3` / `-0.3` | Action bands |
| `AUTO_WITHDRAW_MULTIPLE` / `AUTO_WITHDRAW_FRACTION` | `5.5` / `0.2` | Auto-withdraw rule |
| `DATABASE_URL` | `sqlite:///proverbs.db` | SQLAlchemy URL (Postgres on Railway) |
| `CULTURAL_BACKEND` | `vader` | `vader` or `transformers` |
| `NEWS_BACKEND` | `rss` | `rss` or `newsapi` (+ `NEWSAPI_KEY`) |
| `BROKER` | `paper` | `paper` or `robinhood` |
| `LIVE_TRADING` | `false` | `false` = dry-run (log only); `true` = send real orders |
| `MAX_ORDER_NOTIONAL` | `500` | Hard cap per order (USD) |

---

## HTTP API (dashboard)

`GET /healthz` · `GET /api/signal[?symbol=]` · `GET /api/balance` ·
`GET /api/portfolio` · `GET /api/history` · `GET /api/trading` ·
`GET /api/positions` · `GET /report` (investor report page) ·
`GET /api/report` · `GET /api/accuracy` · `GET /api/leaderboard` ·
`GET /api/watchlist` · `POST /api/deposit` · `POST /api/withdraw` ·
`POST /api/risk` · `POST /api/cycle`

---

## Notes on the rebuild

This is a ground-up rebuild of an earlier five-file prototype. Deliberate
engineering choices for reliability on Railway:
- The deprecated Yahoo CSV download endpoint (now 401/403) → the maintained
  `yfinance` library.
- Fragile homepage `<h3>` scraping → structured RSS feeds (keyless) / NewsAPI.
- Dummy `random.choice` insights and `np.random` sentiment → the real Cultural AI
  index wired into the orchestrator.
- Global in-memory variables that reset on restart → SQLAlchemy persistence.
- `torch`/`prophet`/`xgboost` (heavy; blow past Railway build limits) → VADER +
  scikit-learn by default, with Hugging Face transformers available opt-in.

## Tests
```bash
pip install pytest && python -m pytest -q
```

## Requirements
Outbound internet access (for `yfinance` and RSS). Railway provides this by default.

## License
MIT — see `LICENSE`.
