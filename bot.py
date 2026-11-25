import os
import csv
import time
from datetime import datetime, timedelta, timezone

import numpy as np
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

# ---------- Settings ----------
MARKET_SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD"]
THRESHOLD = 0.60          # buy only when the model is this confident
POSITION_PCT = 0.10       # never use more than 10% of cash
MAX_DAILY_LOSS = 0.02     # stop trading if the account is down 2% today
DRY_RUN = True            # True = only print, no real paper orders
LOG_FILE = "bot_log.csv"

# ---------- Connect ----------
load_dotenv()
trading = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), paper=True)
data = CryptoHistoricalDataClient()

# ---------- Load your trained neuron ----------
model = np.load("model.npz")
w, b, mean, std = model["w"], model["b"], model["mean"], model["std"]


def get_latest_features(symbol):
    """Download recent hourly bars and compute the same 3 features as in training."""
    now = datetime.now(timezone.utc)
    request = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Hour,
        start=now - timedelta(hours=48),
    )
    df = data.get_crypto_bars(request).df.reset_index()

    # Drop the bar that is still in progress. We only use finished hours.
    df = df[df["timestamp"] + timedelta(hours=1) <= now]

    close = df["close"]
    ret_1 = close.pct_change().iloc[-1]
    ret_5 = close.pct_change(5).iloc[-1]
    sma_ratio = (close / close.rolling(20).mean() - 1).iloc[-1]

    features = np.array([ret_1, ret_5, sma_ratio])
    return features, close.iloc[-1], df["timestamp"].iloc[-1]


def predict(features):
    """Normalize exactly like training, then run the neuron."""
    x = (features - mean) / std
    z = x @ w + b
    return 1 / (1 + np.exp(-z))


def position_symbol(symbol):
    return symbol.replace("/", "")


def have_position(symbol):
    try:
        trading.get_open_position(position_symbol(symbol))
        return True
    except Exception:
        return False   # Alpaca raises an error when there is no position


def daily_loss_limit_hit():
    account = trading.get_account()
    change = float(account.equity) / float(account.last_equity) - 1
    return change <= -MAX_DAILY_LOSS


def log(row):
    is_new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["time", "symbol", "bar_time", "price", "p_up", "holding", "action", "dry_run"])
        writer.writerow(row)


def run_once(symbol):
    features, price, bar_time = get_latest_features(symbol)
    p_up = predict(features)
    holding = have_position(symbol)
    action = "HOLD"

    if daily_loss_limit_hit():
        action = "STOP"
        if holding and not DRY_RUN:
            trading.close_position(position_symbol(symbol))

    elif p_up > THRESHOLD and not holding:
        action = "BUY"
        if not DRY_RUN:
            cash = float(trading.get_account().cash)
            order = MarketOrderRequest(
                symbol=symbol,
                notional=round(cash * POSITION_PCT, 2),   # dollar amount, since 1 BTC is expensive
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC,            # crypto requires GTC or IOC
            )
            trading.submit_order(order)

    elif p_up <= THRESHOLD and holding:
        action = "SELL"
        if not DRY_RUN:
            trading.close_position(position_symbol(symbol))

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"{now} | {symbol} | bar {bar_time} | price {price:.2f} | p_up {p_up:.4f} | holding {holding} | {action}")
    log([now, symbol, bar_time, round(price, 2), round(p_up, 4), holding, action, DRY_RUN])
    return action


def seconds_until_next_hour():
    now = datetime.now(timezone.utc)
    next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    return (next_hour - now).total_seconds() + 30   # wait 30s so the new bar exists


# ---------- Main loop ----------
print("Bot started. DRY_RUN =", DRY_RUN, "| Press Ctrl+C to stop.")
while True:
    try:
        stopped = False
        for symbol in MARKET_SYMBOLS:
            action = run_once(symbol)
            if action == "STOP":
                stopped = True
        if stopped:
            print("Daily loss limit hit. Stopping.")
            break
    except Exception as e:
        print("Error:", e)   # keep running even if one hour fails
    wait = seconds_until_next_hour()
    print(f"Sleeping {wait / 60:.1f} minutes...")
    time.sleep(wait)