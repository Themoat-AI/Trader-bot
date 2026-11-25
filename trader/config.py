"""Every setting for the bot lives here, so nothing is copy-pasted between files."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
JOURNAL_DB = ROOT / "data" / "journal.db"

# ---------------------------------------------------------------------------
# Universes: a group of symbols traded with one model.
# Pick which one the bot uses with UNIVERSE (or the TRADER_UNIVERSE env var).
# ---------------------------------------------------------------------------
UNIVERSES = {
    "crypto": {
        "asset_class": "crypto",
        "symbols": ["BTC/USD", "ETH/USD", "SOL/USD"],
        "market_symbol": "BTC/USD",   # "the whole market" feature for every coin
        "fee": 0.0025,                # Alpaca crypto taker fee, per side
        "slippage": 0.0005,
        "bars_per_year": 24 * 365,    # crypto trades every hour of the year
    },
    "stocks": {
        "asset_class": "stock",
        "symbols": ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"],
        "market_symbol": "SPY",       # SPY = the S&P 500, i.e. the market
        "fee": 0.0,                   # Alpaca stock trades are commission-free
        "slippage": 0.0005,
        "bars_per_year": 7 * 252,     # ~7 hourly bars per trading day
    },
}
UNIVERSE = os.getenv("TRADER_UNIVERSE", "crypto")

HISTORY_DAYS = 730      # how much history to download for training

# ---------------------------------------------------------------------------
# What the model predicts
# ---------------------------------------------------------------------------
HORIZON_BARS = 24       # predict the return over the next 24 bars (24h for crypto)
# The target is 1 only if the move beats the round-trip trading cost.
# Predicting "up by more than fees" is what actually matters for profit.

# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------
HIDDEN_LAYERS = [16, 8]
LEARNING_RATE = 1e-3
L2 = 1e-4
EPOCHS = 30
BATCH_SIZE = 256
PATIENCE = 5            # stop early if validation loss hasn't improved for this many epochs
SEED = 42

# Walk-forward testing: train on the past, test on the next block, slide forward.
WALK_FORWARD_FOLDS = 6
WALK_FORWARD_START = 0.4   # first fold trains on the first 40% of history

# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------
ENTER_THRESHOLD = 0.60  # buy when P(beats fees) is above this
EXIT_THRESHOLD = 0.45   # sell when it drops below this (gap avoids flip-flopping)
MIN_HOLD_BARS = 6       # don't sell a fresh position just because of noise
MAX_HOLD_BARS = 48      # always exit after this long (2x the horizon)

# ---------------------------------------------------------------------------
# Risk management (non-negotiable)
# ---------------------------------------------------------------------------
POSITION_PCT = 0.10     # max 10% of equity per position
TARGET_VOL = 0.01       # shrink positions when hourly volatility is above 1%
MAX_POSITIONS = 3
STOP_LOSS = 0.05        # exit a position that is down 5%
MAX_DAILY_LOSS = 0.02   # stop opening trades for the day if equity is down 2%
MAX_DRAWDOWN = 0.15     # shut the bot down if equity falls 15% from its peak

# ---------------------------------------------------------------------------
# Live bot
# ---------------------------------------------------------------------------
DRY_RUN = os.getenv("TRADER_DRY_RUN", "true").lower() != "false"
LOOKBACK_BARS = 400     # recent bars to download each hour (features need ~170)


def universe(name=None):
    """Settings for one universe, plus its name."""
    name = name or UNIVERSE
    if name not in UNIVERSES:
        raise ValueError(f"Unknown universe {name!r}. Choose from {list(UNIVERSES)}")
    return {"name": name, **UNIVERSES[name]}


def round_trip_cost(u):
    return 2 * (u["fee"] + u["slippage"])
