"""Discord bot — the primary interface (build step 8/9 of the plan).

Slash commands let users see live signals and manage their paper account;
a background loop runs the engine every ``CYCLE_INTERVAL_MINUTES`` and posts
alerts (notable BUY/REDUCE calls + auto-withdrawals) to the configured channel.

Engine calls are blocking (network + sklearn), so they run in a worker thread
via ``asyncio.to_thread`` to keep the Discord event loop responsive.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands, tasks

from proverbs.brains.orchestrator import Decision
from proverbs.config import settings
from proverbs.services.engine import CycleReport, engine
from proverbs.trading.manager import trading

logger = logging.getLogger(__name__)

ACTION_COLOR = {
    "BUY": discord.Color.green(),
    "HOLD": discord.Color.gold(),
    "REDUCE": discord.Color.red(),
}


# ------------------------------------------------------------------- embeds
def decision_embed(decision: Decision) -> discord.Embed:
    color = ACTION_COLOR.get(decision.action, discord.Color.light_grey())
    embed = discord.Embed(
        title=f"{decision.emoji} {decision.symbol} — {decision.action}",
        color=color,
    )
    embed.add_field(name="Combined score", value=f"{decision.combined_score:+.3f}", inline=True)
    embed.add_field(name="Confidence", value=f"{decision.confidence*100:.0f}%", inline=True)
    embed.add_field(name="​", value="​", inline=True)
    embed.add_field(name="Fiscal signal", value=f"{decision.fiscal_signal:+.3f}", inline=True)
    embed.add_field(name="Cultural signal", value=f"{decision.cultural_signal:+.3f}", inline=True)
    embed.add_field(name="​", value="​", inline=True)

    if decision.fiscal:
        f = decision.fiscal
        embed.add_field(name="Last price", value=f"${f.last_price:,.2f}", inline=True)
        embed.add_field(name="Sharpe (ann.)", value=f"{f.sharpe_ratio:.2f}", inline=True)
        embed.add_field(name="95% VaR (daily)", value=f"{f.var_95*100:.2f}%", inline=True)
        if f.moving_averages:
            mas = "  ".join(f"{k}=${v:,.2f}" for k, v in f.moving_averages.items())
            embed.add_field(name="Moving averages", value=mas, inline=False)
        embed.set_footer(text=f"model: {f.model}")
    if decision.cultural and decision.cultural.headlines:
        top = "\n".join(f"• {h[:90]}" for h in decision.cultural.headlines[:3])
        embed.add_field(name=f"Cultural read ({decision.cultural.label})", value=top or "—", inline=False)
    return embed


def _account_embed(title: str, data: dict) -> discord.Embed:
    embed = discord.Embed(title=title, color=discord.Color.blurple())
    embed.add_field(name="Balance", value=f"${data.get('balance', 0):,.2f}", inline=True)
    embed.add_field(name="Risk level", value=data.get("risk_level", "—"), inline=True)
    embed.add_field(name="Net profit", value=f"${data.get('profit', 0):,.2f}", inline=True)
    embed.add_field(name="Deposited", value=f"${data.get('total_deposited', 0):,.2f}", inline=True)
    embed.add_field(name="Withdrawn", value=f"${data.get('total_withdrawn', 0):,.2f}", inline=True)
    return embed


def cycle_summary_embed(report: CycleReport) -> discord.Embed:
    """Always-on summary of a cycle (used by the manual /runcycle command)."""
    tag = "LIVE" if trading.state.live else "DRY-RUN"
    broker_name = trading.broker.name if trading.broker else "?"
    embed = discord.Embed(
        title="🔄 Analysis cycle complete",
        description=(f"avg score **{report.avg_score:+.3f}** · sim return "
                     f"**{report.simulated_return_pct:+.3f}%** · balance **${report.balance_after:,.2f}**\n"
                     f"broker **{broker_name}** · mode **{tag}**"),
        color=discord.Color.blurple(),
    )
    for d in report.decisions[:12]:
        embed.add_field(name=f"{d.emoji} {d.symbol} — {d.action}",
                        value=f"score {d.combined_score:+.2f} · conf {d.confidence*100:.0f}%",
                        inline=True)
    acted = [o for o in report.orders if o.status in ("submitted", "dry_run", "rejected")]
    if acted:
        lines = []
        for o in acted[:10]:
            extra = f" — {o.message}" if o.status == "rejected" else ""
            lines.append(f"{o.side.upper()} ${o.notional:,.2f} {o.symbol} · {o.status}{extra}")
        embed.add_field(name=f"⚙️ Orders [{broker_name} · {tag}]",
                        value="\n".join(lines)[:1024], inline=False)
    if report.errors:
        embed.add_field(name="⚠️ Errors", value="\n".join(report.errors[:5])[:1024], inline=False)
    return embed


def cycle_alert_embed(report: CycleReport) -> Optional[discord.Embed]:
    """Build an alert embed for notable events; None if nothing worth pinging."""
    notable = [d for d in report.decisions if d.action in ("BUY", "REDUCE")]
    if not notable and not report.withdrawals and not report.orders:
        return None
    embed = discord.Embed(
        title="📣 proverbs — market alert",
        description=f"Average portfolio score: **{report.avg_score:+.3f}** · "
                    f"simulated return this cycle: **{report.simulated_return_pct:+.3f}%**",
        color=discord.Color.orange(),
    )
    for d in notable[:10]:
        embed.add_field(
            name=f"{d.emoji} {d.symbol} — {d.action}",
            value=f"score {d.combined_score:+.2f} · conf {d.confidence*100:.0f}% "
                  f"· ${d.fiscal.last_price:,.2f}" if d.fiscal else f"score {d.combined_score:+.2f}",
            inline=False,
        )
    for w in report.withdrawals:
        embed.add_field(name="💸 Auto-withdrawal",
                        value=f"${w.amount:,.2f} secured · new balance ${w.balance_after:,.2f}",
                        inline=False)
    acted = [o for o in report.orders if o.status in ("submitted", "dry_run")]
    if acted:
        tag = "DRY-RUN" if not trading.state.live else "LIVE"
        lines = [f"{o.side.upper()} ${o.notional:,.2f} {o.symbol} ({o.status})" for o in acted[:8]]
        embed.add_field(name=f"⚙️ Orders [{trading.broker.name if trading.broker else '?'} · {tag}]",
                        value="\n".join(lines), inline=False)
    return embed


# ------------------------------------------------------------------- bot
def build_bot() -> commands.Bot:
    intents = discord.Intents.default()
    intents.message_content = True  # privileged: enable in the Dev Portal for prefix commands
    bot = commands.Bot(command_prefix=settings.command_prefix, intents=intents,
                       help_command=None)

    # ---- background engine loop ----
    @tasks.loop(minutes=max(1, settings.cycle_interval_minutes))
    async def engine_loop():
        try:
            report = await asyncio.to_thread(engine.run_cycle)
        except Exception:
            logger.exception("engine_loop: cycle failed")
            return
        if settings.alert_channel_id is None:
            return
        channel = bot.get_channel(settings.alert_channel_id)
        if channel is None:
            logger.warning("engine_loop: alert channel %s not found.", settings.alert_channel_id)
            return
        embed = cycle_alert_embed(report)
        if embed is not None:
            try:
                await channel.send(embed=embed)
            except Exception:
                logger.exception("engine_loop: failed to post alert")

    @engine_loop.before_loop
    async def _before_loop():
        await bot.wait_until_ready()

    @bot.event
    async def on_ready():
        logger.info("Discord bot ready as %s (id=%s)", bot.user, bot.user.id if bot.user else "?")
        try:
            if settings.discord_guild_id:
                guild = discord.Object(id=settings.discord_guild_id)
                bot.tree.copy_global_to(guild=guild)
                await bot.tree.sync(guild=guild)
                logger.info("Slash commands synced to guild %s", settings.discord_guild_id)
            else:
                await bot.tree.sync()
                logger.info("Slash commands synced globally (may take up to 1h to appear).")
        except Exception:
            logger.exception("on_ready: command sync failed")
        if not engine_loop.is_running():
            engine_loop.start()

    _register_commands(bot)
    return bot


def _register_commands(bot: commands.Bot) -> None:
    tree = bot.tree

    def uid(interaction: discord.Interaction) -> str:
        return str(interaction.user.id)

    def uname(interaction: discord.Interaction) -> str:
        return interaction.user.display_name

    @tree.command(name="ping", description="Check the bot is alive.")
    async def ping(interaction: discord.Interaction):
        await interaction.response.send_message("🏓 pong — proverbs is running.", ephemeral=True)

    @tree.command(name="signal", description="Show the latest signal(s). Optionally pass a ticker.")
    @app_commands.describe(symbol="Ticker symbol, e.g. AAPL (optional)")
    async def signal(interaction: discord.Interaction, symbol: Optional[str] = None):
        await interaction.response.defer(thinking=True)
        if symbol:
            decision = await asyncio.to_thread(_analyze_symbol, symbol)
            if decision is None:
                await interaction.followup.send(f"Couldn't fetch data for `{symbol.upper()}`.")
                return
            await interaction.followup.send(embed=decision_embed(decision))
        else:
            signals = await asyncio.to_thread(engine.latest_signals)
            if not signals:
                await interaction.followup.send(
                    "No signals yet — the first cycle runs shortly, or try `/analyze <ticker>`.")
                return
            embed = discord.Embed(title="📊 Latest signals", color=discord.Color.blurple())
            for sig in signals:
                emoji = {"BUY": "🟢", "HOLD": "🟡", "REDUCE": "🔴"}.get(sig["action"], "⚪")
                embed.add_field(
                    name=f"{emoji} {sig['symbol']} — {sig['action']}",
                    value=f"score {sig['combined_score']:+.2f} · conf {sig['confidence']*100:.0f}% "
                          f"· ${sig['last_price']:,.2f}",
                    inline=False)
            await interaction.followup.send(embed=embed)

    @tree.command(name="analyze", description="Run a fresh analysis on a ticker right now.")
    @app_commands.describe(symbol="Ticker symbol, e.g. TSLA")
    async def analyze(interaction: discord.Interaction, symbol: str):
        await interaction.response.defer(thinking=True)
        decision = await asyncio.to_thread(_analyze_symbol, symbol)
        if decision is None:
            await interaction.followup.send(f"Couldn't fetch data for `{symbol.upper()}`.")
            return
        await interaction.followup.send(embed=decision_embed(decision))

    @tree.command(name="runcycle", description="Run a full analysis + trade cycle right now (great for testing).")
    async def runcycle(interaction: discord.Interaction):
        await interaction.response.defer(thinking=True)
        report = await asyncio.to_thread(engine.run_cycle)
        await interaction.followup.send(embed=cycle_summary_embed(report))

    @tree.command(name="balance", description="Show your paper account balance.")
    async def balance(interaction: discord.Interaction):
        data = await asyncio.to_thread(engine.get_account, uid(interaction), uname(interaction))
        await interaction.response.send_message(embed=_account_embed("💼 Your account", data))

    @tree.command(name="deposit", description="Deposit into your paper account.")
    @app_commands.describe(amount="Amount to deposit (USD)")
    async def deposit(interaction: discord.Interaction, amount: float):
        res = await asyncio.to_thread(engine.deposit, uid(interaction), amount, uname(interaction))
        if res["status"] != "success":
            await interaction.response.send_message(f"⚠️ {res['message']}", ephemeral=True)
            return
        await interaction.response.send_message(embed=_account_embed("✅ Deposit complete", res))

    @tree.command(name="withdraw", description="Withdraw from your paper account.")
    @app_commands.describe(amount="Amount to withdraw (USD)")
    async def withdraw(interaction: discord.Interaction, amount: float):
        res = await asyncio.to_thread(engine.withdraw, uid(interaction), amount, uname(interaction))
        if res["status"] != "success":
            await interaction.response.send_message(f"⚠️ {res['message']}", ephemeral=True)
            return
        await interaction.response.send_message(embed=_account_embed("✅ Withdrawal complete", res))

    @tree.command(name="risk", description="Set your risk level: Low, Medium, or High.")
    @app_commands.describe(level="Low | Medium | High")
    @app_commands.choices(level=[
        app_commands.Choice(name="Low", value="Low"),
        app_commands.Choice(name="Medium", value="Medium"),
        app_commands.Choice(name="High", value="High"),
    ])
    async def risk(interaction: discord.Interaction, level: app_commands.Choice[str]):
        res = await asyncio.to_thread(engine.set_risk, uid(interaction), level.value, uname(interaction))
        if res["status"] != "success":
            await interaction.response.send_message(f"⚠️ {res['message']}", ephemeral=True)
            return
        await interaction.response.send_message(embed=_account_embed("✅ Risk updated", res))

    @tree.command(name="portfolio", description="Show your account and any positions.")
    async def portfolio(interaction: discord.Interaction):
        data = await asyncio.to_thread(engine.get_portfolio, uid(interaction))
        embed = _account_embed("📁 Your portfolio", data)
        positions = data.get("positions", [])
        if positions:
            lines = [f"{p['symbol']}: {p['quantity']} @ ${p['avg_cost']:,.2f} "
                     f"(now ${p['current_price']:,.2f})" for p in positions]
            embed.add_field(name="Positions", value="\n".join(lines), inline=False)
        else:
            embed.add_field(name="Positions", value="No open positions.", inline=False)
        await interaction.response.send_message(embed=embed)

    @tree.command(name="history", description="Show your recent transactions.")
    async def history(interaction: discord.Interaction):
        txns = await asyncio.to_thread(engine.get_history, uid(interaction), 15)
        if not txns:
            await interaction.response.send_message("No transactions yet.", ephemeral=True)
            return
        embed = discord.Embed(title="🧾 Recent transactions", color=discord.Color.blurple())
        for t in txns:
            embed.add_field(
                name=f"{t['kind']} · ${t['amount']:,.2f}",
                value=f"{t['timestamp'][:19].replace('T', ' ')} · balance ${t['balance_after']:,.2f}",
                inline=False)
        await interaction.response.send_message(embed=embed)

    @tree.command(name="watchlist", description="Show the tickers proverbs is tracking.")
    async def watchlist(interaction: discord.Interaction):
        await interaction.response.send_message(
            "👀 Watchlist: " + ", ".join(f"`{s}`" for s in settings.watchlist))

    # ---------------------------------------------------------- trading cmds
    @tree.command(name="trading", description="Show broker status, mode, and risk limits.")
    async def trading_status(interaction: discord.Interaction):
        status = await asyncio.to_thread(trading.status)
        mode = status["mode"]
        color = discord.Color.red() if mode == "LIVE" else discord.Color.greyple()
        embed = discord.Embed(title=f"⚙️ Trading — {status['broker']} · {mode}", color=color)
        embed.add_field(name="Trading enabled", value="✅" if status["trading_enabled"] else "🛑 halted", inline=True)
        embed.add_field(name="Mode", value=mode + (" 💵" if mode == "LIVE" else " (safe)"), inline=True)
        acc = status.get("account")
        if acc:
            embed.add_field(name="Equity", value=f"${acc['equity']:,.2f}", inline=True)
            embed.add_field(name="Cash", value=f"${acc['cash']:,.2f}", inline=True)
            embed.add_field(name="Buying power", value=f"${acc['buying_power']:,.2f}", inline=True)
        lim = status["limits"]
        embed.add_field(
            name="Limits",
            value=(f"order ${lim['base_order_notional']:.0f}–${lim['max_order_notional']:.0f} · "
                   f"max/pos ${lim['max_position_notional']:.0f} · "
                   f"min conf {lim['order_confidence_min']:.2f} · "
                   f"daily loss ${lim['daily_loss_limit']:.0f}"),
            inline=False)
        embed.set_footer(text="Change with /kill, /resume, /mode. Set LIVE only when you mean it.")
        await interaction.response.send_message(embed=embed)

    @tree.command(name="positions", description="Show current broker positions.")
    async def positions(interaction: discord.Interaction):
        await interaction.response.defer(thinking=True)
        if trading.broker is None:
            await interaction.followup.send("No broker connected.")
            return
        pos = await asyncio.to_thread(trading.broker.get_positions)
        if not pos:
            await interaction.followup.send("No open positions.")
            return
        embed = discord.Embed(title=f"📊 Positions ({trading.broker.name})", color=discord.Color.blurple())
        for p in pos[:20]:
            embed.add_field(
                name=f"{p.symbol} — {p.quantity:.4f}",
                value=f"avg ${p.avg_cost:,.2f} · now ${p.current_price:,.2f} · "
                      f"value ${p.market_value:,.2f} · P/L ${p.unrealized_pl:,.2f}",
                inline=False)
        await interaction.followup.send(embed=embed)

    @tree.command(name="order", description="Place a manual order (respects dry-run + risk limits).")
    @app_commands.describe(side="buy or sell", symbol="Ticker, e.g. AAPL", notional="USD amount")
    @app_commands.choices(side=[
        app_commands.Choice(name="buy", value="buy"),
        app_commands.Choice(name="sell", value="sell"),
    ])
    async def order(interaction: discord.Interaction, side: app_commands.Choice[str],
                    symbol: str, notional: float):
        await interaction.response.defer(thinking=True)
        result = await asyncio.to_thread(trading.manual_order, symbol, side.value, notional)
        emoji = {"submitted": "✅", "dry_run": "🧪", "rejected": "⛔", "error": "⚠️"}.get(result.status, "•")
        await interaction.followup.send(
            f"{emoji} **{result.status.upper()}** — {side.value} ${result.notional:,.2f} "
            f"{result.symbol}" + (f" @ ${result.price:,.2f}" if result.price else "") +
            f"\n{result.message}")

    @tree.command(name="kill", description="Halt all trading immediately (kill switch).")
    async def kill(interaction: discord.Interaction):
        trading.halt()
        await interaction.response.send_message("🛑 Trading **halted**. No orders will be placed until `/resume`.")

    @tree.command(name="resume", description="Resume trading after a halt.")
    async def resume(interaction: discord.Interaction):
        trading.resume()
        await interaction.response.send_message("✅ Trading **resumed**.")

    @tree.command(name="mode", description="Switch between dry-run and LIVE trading.")
    @app_commands.describe(mode="dry-run (safe) or live (real orders)")
    @app_commands.choices(mode=[
        app_commands.Choice(name="dry-run", value="dry"),
        app_commands.Choice(name="live", value="live"),
    ])
    async def mode(interaction: discord.Interaction, mode: app_commands.Choice[str]):
        if mode.value == "live":
            trading.set_live(True)
            await interaction.response.send_message(
                "💵 **LIVE mode enabled.** Real orders will now be placed on "
                f"`{trading.broker.name if trading.broker else settings.broker}` within your risk limits. "
                "Use `/kill` to stop instantly.", ephemeral=False)
        else:
            trading.set_live(False)
            await interaction.response.send_message("🧪 **Dry-run mode.** Orders are logged but not sent.")

    @tree.command(name="help", description="What proverbs can do.")
    async def help_cmd(interaction: discord.Interaction):
        embed = discord.Embed(
            title="🤖 proverbs — commands",
            description="A three-brain paper-trading signal bot. **Simulation only — not financial advice.**",
            color=discord.Color.blurple())
        embed.add_field(name="Signals",
                        value="`/signal [ticker]` · `/analyze <ticker>` · `/runcycle` · `/watchlist`", inline=False)
        embed.add_field(name="Account",
                        value="`/balance` · `/deposit <amt>` · `/withdraw <amt>` · "
                              "`/risk <level>` · `/portfolio` · `/history`", inline=False)
        embed.add_field(name="Trading",
                        value="`/trading` · `/positions` · `/order <buy|sell> <ticker> <amt>` · "
                              "`/mode <dry-run|live>` · `/kill` · `/resume`", inline=False)
        embed.set_footer(text="Trading defaults to DRY-RUN. Alerts post to the configured channel.")
        await interaction.response.send_message(embed=embed, ephemeral=True)


def _analyze_symbol(symbol: str) -> Optional[Decision]:
    """Blocking helper: full analysis for one symbol (run in a worker thread)."""
    fiscal = engine.fiscal.analyze(symbol)
    cultural = engine.cultural.analyze(symbol)
    if fiscal is None and cultural is None:
        return None
    decision = engine.orchestrator.decide(fiscal, cultural, symbol)
    try:
        from proverbs.persistence import db

        with db.session_scope() as s:
            db.save_signal(s, decision)
    except Exception:
        logger.exception("_analyze_symbol: failed to persist signal for %s", symbol)
    return decision


async def start_bot() -> None:
    """Async entrypoint used by main.py."""
    if not settings.discord_token:
        raise RuntimeError("DISCORD_TOKEN is not set; cannot start the Discord bot.")
    bot = build_bot()
    async with bot:
        await bot.start(settings.discord_token)
