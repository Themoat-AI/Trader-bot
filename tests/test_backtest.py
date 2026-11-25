import numpy as np
import pandas as pd
import pytest

from trader.backtest import simulate
from trader.risk import RiskLimits
from trader.strategy import Params

U = {"fee": 0.0025, "slippage": 0.0, "bars_per_year": 8760}


def preds(prices, p):
    t = pd.date_range("2025-01-01", periods=len(prices), freq="1h", tz="UTC")
    return pd.DataFrame({"timestamp": t, "symbol": "BTC/USD", "close": prices, "p_up": p, "vol_24": 0.005})


def test_flat_price_round_trip_costs_two_fees():
    df = preds(np.full(20, 100.0), [0.9] + [0.1] * 19)
    res = simulate(df, U, Params(min_hold_bars=1), RiskLimits(position_pct=0.10))
    assert len(res.trades) == 1
    # $10,000 position, 0.25% in and 0.25% out
    assert res.equity.iloc[-1] == pytest.approx(100_000 - 25 - 9_975 * 0.0025, rel=1e-9)


def test_profits_when_price_rises_and_model_is_in():
    df = preds(np.linspace(100, 120, 30), [0.9] * 30)
    res = simulate(df, U, Params(max_hold_bars=100), RiskLimits(position_pct=0.10))
    assert res.metrics["strategy_total_return"] > 0.015


def test_no_trades_without_signal():
    df = preds(np.linspace(100, 50, 30), [0.3] * 30)
    res = simulate(df, U)
    assert res.trades.empty and res.equity.iloc[-1] == 100_000
    assert res.metrics["buy_hold_total_return"] == pytest.approx(-0.5)


def test_stop_loss_limits_loss():
    prices = np.r_[100, 100, 90, 80, 70, 60]
    df = preds(prices, [0.9] * 6)
    res = simulate(df, U, Params(stop_loss=0.05), RiskLimits(position_pct=0.10))
    assert res.trades.iloc[0]["reason"] == "stop loss"
    assert res.trades.iloc[0]["exit_price"] == 90
