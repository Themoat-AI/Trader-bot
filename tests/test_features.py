import numpy as np
import pandas as pd

from tests.conftest import make_bars
from trader.features import FEATURE_COLUMNS, add_target, make_features, regularize


def test_no_lookahead(bars):
    """Features at time t must not change when we delete all data after t."""
    full = make_features(bars, "BTC/USD")
    cutoff = bars["timestamp"].sort_values().unique()[300]
    partial = make_features(bars[bars["timestamp"] <= cutoff], "BTC/USD")
    a = full[full["timestamp"] <= cutoff].reset_index(drop=True)[FEATURE_COLUMNS]
    b = partial.reset_index(drop=True)[FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(a, b)


def test_features_exist_after_warmup(bars):
    df = make_features(bars, "BTC/USD").dropna(subset=FEATURE_COLUMNS)
    assert len(df) > 0
    assert set(FEATURE_COLUMNS) <= set(df.columns)
    assert np.isfinite(df[FEATURE_COLUMNS].to_numpy()).all()


def test_market_feature_matches_market_symbol(bars):
    df = make_features(bars, "BTC/USD").dropna(subset=FEATURE_COLUMNS)
    btc = df[df["symbol"] == "BTC/USD"]
    np.testing.assert_allclose(btc["mkt_ret_24"], btc["ret_24"])


def test_target_is_future_return(bars):
    df = add_target(make_features(bars, "BTC/USD"), horizon=24, hurdle=0.0)
    btc = df[df["symbol"] == "BTC/USD"].reset_index(drop=True)
    expected = btc["close"].iloc[24] / btc["close"].iloc[0] - 1
    assert np.isclose(btc["future_ret"].iloc[0], expected)
    assert btc["target"].iloc[-24:].isna().all()     # no future price -> no fake label
    assert btc["target"].iloc[0] == float(expected > 0)


def test_target_never_crosses_symbols(bars):
    df = add_target(make_features(bars, "BTC/USD"), horizon=5, hurdle=0.0)
    for _, g in df.groupby("symbol"):
        assert g["future_ret"].iloc[-5:].isna().all()


def test_regularize_fills_missing_hours():
    bars = make_bars(symbols=("BTC/USD",), n=50)
    gappy = bars.drop(index=[10, 11, 12])
    out = regularize(gappy)
    assert len(out) == 50
    assert out.loc[10, "close"] == out.loc[9, "close"]
    assert out.loc[10, "volume"] == 0
