# Trader bot

A learning project: a neural network written from scratch in NumPy that paper-trades
crypto (and optionally stocks) on Alpaca, with walk-forward testing, a portfolio
backtester, risk management, a SQLite trade journal and a live dashboard.

**Paper trading only.** The backtest results below are honest: so far the model has
**no proven edge**. The point is a correct, well-tested system you can experiment with.

## How it works

```
Alpaca bars ─► features.py ─► nn.py (MLP, from scratch) ─► strategy.py ─► risk.py ─► broker.py
                  ▲                                            ▲              ▲
                  └──── the same code is used by train.py, backtest.py and bot.py ────┘
```

| File | What it does |
|---|---|
| `trader/config.py` | Every setting: symbols, fees, model size, thresholds, risk limits |
| `trader/data.py` | Downloads hourly bars from Alpaca and caches them in `data/` |
| `trader/features.py` | The **only** place features are computed (13 features, no lookahead) |
| `trader/nn.py` | Neural network from scratch: Dense, ReLU, backprop, Adam, early stopping, gradient check |
| `trader/train.py` | Walk-forward training + final model; compares the MLP with logistic regression |
| `trader/strategy.py` | Buy/sell rule shared by the backtest and the live bot |
| `trader/risk.py` | Position sizing by volatility, max positions, daily loss limit, drawdown kill switch |
| `trader/backtest.py` | Hour-by-hour portfolio simulation with fees + slippage, metrics and an equity chart |
| `trader/broker.py` | Alpaca paper-trading wrapper |
| `trader/journal.py` | SQLite log of every decision, order and equity snapshot |
| `trader/bot.py` | The live bot |
| `trader/dashboard.py` | Streamlit dashboard |
| `trader/report.py` | Markdown summary of the journal (shown on GitHub Actions runs) |
| `tests/` | 23 tests: no-lookahead, gradient check, XOR, fee math, risk rules, bot cycle |
| `learning/` | The original day-one scripts |
| `deploy/` | State scripts for GitHub Actions, plus a systemd service + retrain script for your own server |
| `.github/workflows/` | Hourly trading, weekly retraining and tests on GitHub Actions |

### What the model predicts
"Will the next **24 hours** return beat the **round-trip trading cost**?" The day-one
model predicted the next 1 hour, where moves are smaller than fees, so it couldn't
profit even when it was right.

### Strategy and risk rules
- Buy when P > 0.40, sell when P < 0.33 (after holding at least 6h), always exit after 48h.
  (0.60 / 0.45 trades far less and lost less in the backtest.)
- 5% stop loss per position.
- Max 10% of equity per position, scaled down for volatile assets; max 3 positions.
- Stop opening trades for the day at -2%; shut down completely at -15% from peak.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` (it's in `.gitignore`, **never commit it**):
```
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
```

## Usage

```bash
python -m pytest                        # run the tests
python -m trader.data                   # download 2 years of hourly bars
python -m trader.train                  # walk-forward test + train the final model
python -m trader.backtest               # simulate trading on out-of-sample predictions
python -m trader.backtest --model logistic
python -m trader.bot --once             # one dry-run cycle (no orders)
python -m trader.bot                    # dry run, every hour
python -m trader.bot --live             # place PAPER orders every hour
streamlit run trader/dashboard.py       # dashboard
```

Use `--universe stocks` (SPY, QQQ, AAPL, MSFT, NVDA) to trade stocks. The bot skips
cycles while the market is closed.

## Results so far (crypto, Sep 2025 – Nov 2026, out-of-sample, after costs)

| | MLP [16, 8] | Logistic (1 neuron) | Buy & hold |
|---|---|---|---|
| Mean AUC (0.5 = coin flip) | 0.508 | 0.500 | — |
| Total return | -11.4% | -4.7% | -62.3% |
| Max drawdown | -12.7% | -6.5% | -64.8% |
| Trades | 136 | 34 | — |

It lost far less than buy & hold, but only because it sat in cash ~93% of the time
during a crash. The predictions themselves are close to a coin flip. Ideas to try next
are listed below.

## Ideas to improve the edge
- Different horizons (`HORIZON_BARS` = 6, 48, 168) and thresholds. Change one thing at a
  time and always judge it on walk-forward results, never on a single lucky run.
- New features: funding rates, order-book imbalance, cross-asset signals, day-of-week.
- Longer history, or daily bars instead of hourly.
- Compare everything against the logistic baseline: a bigger network has to earn its keep.

## Running 24/7 on GitHub (no computer needed)
GitHub Actions runs the bot for free from this repo:

| Workflow | When | What |
|---|---|---|
| **Trade** | every 15 min (acts once per hourly bar) | one bot cycle, placing **paper** orders; summary shown on the run page |
| **Retrain** | Sundays 03:23 UTC | fresh data, retrain, backtest |
| **Tests** | every push | runs the test suite |

The journal, model and charts are saved to the `bot-state` branch between runs.

**Setup (once):** repo **Settings → Secrets and variables → Actions**
1. **Secrets** tab → add `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` (your **paper** keys).
2. **Actions** tab → **Trade** → **Run workflow** to start it right away.
3. It places **paper** orders by default. To only log decisions without ordering:
   **Variables** tab → add `TRADER_DRY_RUN` = `true` (delete it to resume trading).

Notes: GitHub can start scheduled runs a few minutes late, and it pauses schedules in
repos with no activity for 60 days (re-enable them on the Actions tab).

## Running on your own server instead
See `deploy/trader-bot.service` (auto-restart on crash) and `deploy/retrain.sh`
(monthly retrain). A small Linux VPS or a Raspberry Pi is enough.
