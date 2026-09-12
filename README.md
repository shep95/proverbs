# 📜 proverbs

A three-brain algorithmic **investment signal bot** for Discord, hosted on Railway.

`proverbs` reads market *culture* (news sentiment), scores the *quantitative*
risk/reward of each ticker, and an *orchestrator* fuses the two into a
**BUY / HOLD / REDUCE** call. It posts scheduled alerts to a Discord channel and
answers slash commands, and it ships a live web dashboard as a secondary view.

> ⚠️ **Paper-trading simulation — not financial advice.** `proverbs` never places
> real brokerage orders. Balances, growth, and auto-withdrawals are simulated for
> education and experimentation.

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
| `/balance` | Your paper account balance & risk level |
| `/deposit <amount>` | Deposit into your paper account |
| `/withdraw <amount>` | Withdraw from your paper account |
| `/risk <Low\|Medium\|High>` | Set your risk level |
| `/portfolio` | Account + any positions |
| `/history` | Recent transactions |
| `/watchlist` | Tickers being tracked |
| `/help` · `/ping` | Help / liveness |

Scheduled alerts (notable BUY/REDUCE calls + auto-withdrawals) post automatically
to `DISCORD_ALERT_CHANNEL_ID` every `CYCLE_INTERVAL_MINUTES`.

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

---

## HTTP API (dashboard)

`GET /healthz` · `GET /api/signal[?symbol=]` · `GET /api/balance` ·
`GET /api/portfolio` · `GET /api/history` · `POST /api/deposit` ·
`POST /api/withdraw` · `POST /api/risk` · `POST /api/cycle`

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
