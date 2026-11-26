"""The live paper-trading bot.

    python -m trader.bot              # dry run: decides and logs, places no orders
    python -m trader.bot --live       # place PAPER orders on Alpaca
    python -m trader.bot --once       # run one cycle and exit (good for testing / cron)

Every hour, just after the bar closes:
  1. Check the market is open (stocks) and read the account + positions from Alpaca.
  2. Download recent bars and compute features with the SAME code as training.
  3. Predict P(next 24h return beats fees) with the trained network.
  4. Apply strategy.decide() and the risk rules — the same ones the backtest used.
  5. Place orders, and log everything to the SQLite journal.
"""
import argparse
import logging
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from trader import config, nn
from trader.data import drop_unfinished, fetch_bars
from trader.features import FEATURE_COLUMNS, make_features, regularize
from trader.journal import Journal
from trader.risk import RiskLimits, RiskState, position_size
from trader.strategy import Params, decide

log = logging.getLogger("trader")


def latest_features(u, fetch=fetch_bars, now=None):
    """Most recent finished bar for every symbol, with features."""
    now = now or datetime.now(timezone.utc)
    # Stocks only have ~7 hourly bars a day, so look further back to get enough of them.
    days = config.LOOKBACK_BARS / 24 if u["asset_class"] == "crypto" else config.LOOKBACK_BARS / 7 * 1.5
    bars = drop_unfinished(fetch(u, now - timedelta(days=days)), now)
    if u["asset_class"] == "crypto":
        bars = regularize(bars)
    df = make_features(bars, u["market_symbol"])
    latest = df.dropna(subset=FEATURE_COLUMNS).groupby("symbol").tail(1)
    return latest.set_index("symbol"), bars


def bars_since(bars, symbol, entry_time):
    if entry_time is None:
        return config.MIN_HOLD_BARS   # unknown (e.g. bought by hand): allow normal exits
    t = pd.Timestamp(entry_time)
    return int(((bars["symbol"] == symbol) & (bars["timestamp"] > t)).sum())


def run_cycle(u, model, scaler, broker, journal, risk, params=Params(), limits=RiskLimits(),
              fetch=fetch_bars, now=None):
    """One full decision cycle. Returns a list of (symbol, action, reason)."""
    if not broker.market_open():
        log.info("Market closed, skipping.")
        return []

    acct = broker.account()
    positions = broker.positions()
    # Alpaca's last_equity = equity at the previous close: our "start of day".
    today = (now or datetime.now(timezone.utc)).date()
    if risk.day != today:
        risk.day, risk.day_start_equity = today, acct["last_equity"] or acct["equity"]
    risk.update(acct["equity"], today)

    latest, bars = latest_features(u, fetch, now)
    missing = [s for s in u["symbols"] if s not in latest.index]
    if missing:
        log.warning("No fresh features for %s", missing)
    probs = pd.Series(model.predict_proba(scaler.transform(latest[FEATURE_COLUMNS].to_numpy())),
                      index=latest.index)
    actions = []

    # --- exits ---
    for symbol, pos in list(positions.items()):
        if symbol not in latest.index:
            continue
        price, p = float(latest.at[symbol, "close"]), float(probs[symbol])
        held = bars_since(bars, symbol, journal.entry_time(symbol))
        action, reason = decide(p, True, held, pos["avg_entry_price"], price, params)
        if risk.killed:
            action, reason = "SELL", "max drawdown kill switch"
        journal.decision(symbol, latest.at[symbol, "timestamp"], price, p, True, action, reason, broker.dry_run)
        if action == "SELL":
            order_id = broker.sell(symbol)
            journal.order(symbol, "sell", pos["market_value"], price, order_id, reason, broker.dry_run)
            if not broker.dry_run:
                journal.clear_entry(symbol)
                del positions[symbol]
        actions.append((symbol, action, reason))
        log.info("%-8s p_up %.3f held %3d bars  %s (%s)", symbol, p, held, action, reason)

    # --- entries, most confident first ---
    open_count = len(positions)
    cash = acct["cash"]
    for symbol in probs.sort_values(ascending=False).index:
        if symbol in positions:
            continue
        price, p = float(latest.at[symbol, "close"]), float(probs[symbol])
        action, reason = decide(p, False, params=params)
        if action == "BUY":
            allowed, why = risk.can_open(acct["equity"], open_count)
            dollars = position_size(acct["equity"], cash, float(latest.at[symbol, "vol_24"]), limits)
            if not allowed:
                action, reason = "HOLD", f"blocked: {why}"
            elif dollars < 10:
                action, reason = "HOLD", "position too small"
            else:
                order_id = broker.buy(symbol, dollars)
                journal.order(symbol, "buy", dollars, price, order_id, reason, broker.dry_run)
                if not broker.dry_run:
                    journal.set_entry(symbol, latest.at[symbol, "timestamp"])
                open_count += 1
                cash -= dollars
        journal.decision(symbol, latest.at[symbol, "timestamp"], price, p, False, action, reason, broker.dry_run)
        actions.append((symbol, action, reason))
        log.info("%-8s p_up %.3f              %s (%s)", symbol, p, action, reason)

    journal.snapshot(acct["equity"], acct["cash"], len(positions))
    return actions


def last_finished_bar(now=None):
    """Start time of the most recent completed hourly bar."""
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    return now.floor("h") - pd.Timedelta(hours=1)


def seconds_until_next_bar(now=None):
    now = now or datetime.now(timezone.utc)
    nxt = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    return (nxt - now).total_seconds() + 60   # give the data provider a minute to publish the bar


def load_model(u):
    path = config.MODEL_DIR / f"{u['name']}.npz"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python -m trader.train --universe {u['name']}")
    model, scaler, meta = nn.load(path)
    if meta["features"] != FEATURE_COLUMNS:
        raise SystemExit("Model was trained with different features. Retrain: python -m trader.train")
    return model, scaler, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default=config.UNIVERSE)
    parser.add_argument("--live", action="store_true", help="place paper orders (default: dry run)")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    config.DATA_DIR.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(config.DATA_DIR / "bot.log")])

    from trader.broker import AlpacaBroker

    u = config.universe(args.universe)
    dry_run = config.DRY_RUN and not args.live
    model, scaler, meta = load_model(u)
    broker = AlpacaBroker(u, dry_run=dry_run)
    journal = Journal(config.JOURNAL_DB, u["name"])
    risk = RiskState()
    risk.peak_equity = journal.peak_equity()   # survive restarts without forgetting the peak

    log.info("Bot started: universe=%s symbols=%s dry_run=%s model trained until %s",
             u["name"], u["symbols"], dry_run, meta["trained_until"])
    if args.once and journal.has_bar(last_finished_bar()):
        # Scheduled runs fire every 15 minutes in case GitHub skips some; only the
        # first one after each hourly bar closes actually trades.
        log.info("Bar %s already handled, nothing to do.", last_finished_bar())
        return

    errors = 0
    while True:
        try:
            run_cycle(u, model, scaler, broker, journal, risk)
            errors = 0
        except Exception:
            errors += 1
            log.exception("Cycle failed (%d in a row)", errors)
        if risk.killed:
            log.error("Max drawdown kill switch triggered. Bot stopped — review before restarting.")
            break
        if args.once:
            break
        wait = seconds_until_next_bar() if errors < 5 else 900
        log.info("Sleeping %.1f minutes", wait / 60)
        time.sleep(wait)


if __name__ == "__main__":
    main()
