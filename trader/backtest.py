"""Portfolio backtester: replays the walk-forward predictions hour by hour.

    python -m trader.backtest                     # uses the MLP predictions
    python -m trader.backtest --model logistic    # compare with the 1-neuron model
    python -m trader.backtest --enter 0.65 --exit 0.5

Uses the exact same strategy.decide() and risk rules as the live bot, with
fees and slippage on every trade.
"""
import argparse
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trader import config
from trader.risk import RiskLimits, RiskState, position_size
from trader.strategy import Params, decide


@dataclass
class Result:
    equity: pd.Series
    benchmark: pd.Series
    trades: pd.DataFrame
    exposure: pd.Series
    fees: float = 0.0
    metrics: dict = field(default_factory=dict)


def simulate(preds, u, params=Params(), limits=RiskLimits(), start_cash=100_000.0):
    close = preds.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    p_up = preds.pivot(index="timestamp", columns="symbol", values="p_up").reindex(close.index)
    vol = preds.pivot(index="timestamp", columns="symbol", values="vol_24").reindex(close.index)
    last_price = close.ffill()
    symbols = list(close.columns)
    cost = u["fee"] + u["slippage"]   # charged on each side of a trade

    cash, fees = start_cash, 0.0
    positions = {}   # symbol -> dict(qty, entry_price, basis, bars_held, entry_time)
    risk = RiskState(limits)
    equity_curve, exposure, trades = [], [], []

    for i, t in enumerate(close.index):
        prices = last_price.iloc[i]
        equity = cash + sum(pos["qty"] * prices[s] for s, pos in positions.items())
        risk.update(equity, t.date())

        # --- exits first, so freed cash can be reused this hour ---
        for s in list(positions):
            pos = positions[s]
            pos["bars_held"] += 1
            price, p = close.iat[i, symbols.index(s)], p_up.iat[i, symbols.index(s)]
            if np.isnan(price) or np.isnan(p):
                continue
            action, reason = decide(p, True, pos["bars_held"], pos["entry_price"], price, params)
            if risk.killed:
                action, reason = "SELL", "max drawdown kill switch"
            if action == "SELL":
                gross = pos["qty"] * price
                fee = gross * cost
                cash += gross - fee
                fees += fee
                trades.append({"symbol": s, "entry_time": pos["entry_time"], "exit_time": t,
                               "bars_held": pos["bars_held"], "entry_price": pos["entry_price"],
                               "exit_price": price, "pnl": gross - fee - pos["basis"],
                               "return": (gross - fee) / pos["basis"] - 1, "reason": reason})
                del positions[s]

        # --- entries: most confident symbols first ---
        candidates = sorted((s for s in symbols if s not in positions),
                            key=lambda s: -np.nan_to_num(p_up.iat[i, symbols.index(s)], nan=-1))
        for s in candidates:
            j = symbols.index(s)
            price, p, v = close.iat[i, j], p_up.iat[i, j], vol.iat[i, j]
            if np.isnan(price) or np.isnan(p):
                continue
            if decide(p, False, params=params)[0] != "BUY":
                break   # sorted by confidence, so nobody after this qualifies either
            allowed, _ = risk.can_open(equity, len(positions))
            if not allowed:
                break
            dollars = position_size(equity, cash, v, limits)
            if dollars < 10:
                continue
            fee = dollars * cost
            positions[s] = {"qty": (dollars - fee) / price, "entry_price": price, "basis": dollars,
                            "bars_held": 0, "entry_time": t}
            cash -= dollars
            fees += fee

        equity_curve.append(cash + sum(pos["qty"] * prices[s] for s, pos in positions.items()))
        exposure.append(sum(pos["qty"] * prices[s] for s, pos in positions.items()) / equity_curve[-1])

    equity = pd.Series(equity_curve, index=close.index, name="strategy")
    growth = last_price.bfill() / last_price.bfill().iloc[0]
    benchmark = (growth.mean(axis=1) * start_cash).rename("buy_and_hold")
    result = Result(equity, benchmark, pd.DataFrame(trades), pd.Series(exposure, index=close.index), fees)
    result.metrics = metrics(result, u["bars_per_year"])
    return result


def curve_stats(equity, bars_per_year):
    r = equity.pct_change().dropna()
    years = len(equity) / bars_per_year
    sharpe = r.mean() / r.std() * np.sqrt(bars_per_year) if r.std() > 0 else 0.0
    return {
        "total_return": equity.iloc[-1] / equity.iloc[0] - 1,
        "cagr": (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan,
        "sharpe": sharpe,
        "max_drawdown": (equity / equity.cummax() - 1).min(),
    }


def metrics(res, bars_per_year):
    m = {f"strategy_{k}": v for k, v in curve_stats(res.equity, bars_per_year).items()}
    m.update({f"buy_hold_{k}": v for k, v in curve_stats(res.benchmark, bars_per_year).items()})
    t = res.trades
    m["trades"] = len(t)
    m["win_rate"] = (t["pnl"] > 0).mean() if len(t) else np.nan
    m["avg_trade_return"] = t["return"].mean() if len(t) else np.nan
    m["avg_bars_held"] = t["bars_held"].mean() if len(t) else np.nan
    m["time_in_market"] = (res.exposure > 0).mean()
    m["avg_exposure"] = res.exposure.mean()
    m["fees_paid"] = res.fees
    return m


def plot(res, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(res.equity.index, res.equity, label="Strategy", lw=1.5)
    ax1.plot(res.benchmark.index, res.benchmark, label="Buy & hold (equal weight)", lw=1, alpha=0.8)
    ax1.set_ylabel("Equity ($)")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(alpha=0.3)
    dd = res.equity / res.equity.cummax() - 1
    ax2.fill_between(dd.index, dd * 100, 0, alpha=0.4)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def print_metrics(m):
    pct = {"total_return", "cagr", "max_drawdown", "win_rate", "avg_trade_return",
           "time_in_market", "avg_exposure"}
    rows = [("", "Strategy", "Buy & hold")]
    for k in ("total_return", "cagr", "sharpe", "max_drawdown"):
        a, b = m[f"strategy_{k}"], m[f"buy_hold_{k}"]
        fmt = (lambda x: f"{x:8.2%}") if k in pct else (lambda x: f"{x:8.2f}")
        rows.append((k, fmt(a), fmt(b)))
    for k, a, b in rows:
        print(f"  {k:<16}{a:>10}{b:>12}")
    for k in ("trades", "win_rate", "avg_trade_return", "avg_bars_held", "time_in_market",
              "avg_exposure", "fees_paid"):
        v = m[k]
        if isinstance(v, float) and np.isnan(v):
            s = "n/a"
        elif k in pct:
            s = f"{v:.2%}"
        elif k == "fees_paid":
            s = f"${v:,.0f}"
        else:
            s = f"{v:.1f}" if isinstance(v, float) else str(v)
        print(f"  {k:<16}{s:>10}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default=config.UNIVERSE)
    parser.add_argument("--model", default="mlp", choices=["mlp", "logistic"])
    parser.add_argument("--enter", type=float, default=config.ENTER_THRESHOLD)
    parser.add_argument("--exit", type=float, default=config.EXIT_THRESHOLD)
    args = parser.parse_args()

    u = config.universe(args.universe)
    path = config.REPORT_DIR / f"{u['name']}_oos_{args.model}.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python -m trader.train --universe {u['name']}")
    preds = pd.read_csv(path)
    preds["timestamp"] = pd.to_datetime(preds["timestamp"], utc=True)

    params = Params(enter_threshold=args.enter, exit_threshold=args.exit)
    res = simulate(preds, u, params)
    start, end = res.equity.index[0].date(), res.equity.index[-1].date()
    print(f"Backtest {u['name']} / {args.model}  {start} to {end}  "
          f"(enter > {args.enter}, exit < {args.exit}, cost {u['fee'] + u['slippage']:.2%}/side)")
    print_metrics(res.metrics)

    out = config.REPORT_DIR / f"{u['name']}_equity_{args.model}.png"
    plot(res, out, f"{u['name']} — {args.model} (out-of-sample, after costs)")
    res.trades.to_csv(config.REPORT_DIR / f"{u['name']}_trades_{args.model}.csv", index=False)
    print(f"Chart: {out}")


if __name__ == "__main__":
    main()
