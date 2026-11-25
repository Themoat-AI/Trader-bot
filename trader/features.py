"""Feature engineering — the ONE place features are computed.

Training, backtesting and the live bot all call make_features(), so they can
never disagree. Every feature at row t uses only data from row t or earlier.
"""
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "ret_1", "ret_6", "ret_24",
    "sma_ratio_24", "sma_ratio_168",
    "vol_24", "vol_ratio",
    "rsi_14", "volume_z",
    "hour_sin", "hour_cos",
    "mkt_ret_24", "mkt_sma_ratio_168",
]


def regularize(bars, freq="1h"):
    """Fill missing hours so 'shift by 24 rows' really means 24 hours.

    Crypto sometimes has hours with no trades. We carry the last price forward
    and set volume to 0. Only used for 24/7 markets.
    """
    out = []
    for symbol, g in bars.groupby("symbol"):
        g = g.set_index("timestamp").sort_index()
        full = pd.date_range(g.index[0], g.index[-1], freq=freq)
        g = g.reindex(full)
        g["close"] = g["close"].ffill()
        for col in ("open", "high", "low"):
            g[col] = g[col].fillna(g["close"])
        g["volume"] = g["volume"].fillna(0.0)
        g["symbol"] = symbol
        out.append(g.rename_axis("timestamp").reset_index())
    return pd.concat(out, ignore_index=True)


def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0)


def _symbol_features(g):
    c = g["close"]
    log_ret = np.log(c).diff()
    f = pd.DataFrame(index=g.index)
    f["ret_1"] = c.pct_change()
    f["ret_6"] = c.pct_change(6)
    f["ret_24"] = c.pct_change(24)
    f["sma_ratio_24"] = c / c.rolling(24).mean() - 1
    f["sma_ratio_168"] = c / c.rolling(168).mean() - 1        # 168 hours = 1 week
    f["vol_24"] = log_ret.rolling(24).std()
    f["vol_ratio"] = f["vol_24"] / log_ret.rolling(168).std()
    f["rsi_14"] = rsi(c) / 100 - 0.5                           # scale to -0.5..0.5
    v = np.log1p(g["volume"])
    f["volume_z"] = (v - v.rolling(168).mean()) / v.rolling(168).std()
    return f


def make_features(bars, market_symbol):
    """bars: columns symbol, timestamp, open, high, low, close, volume.

    Returns the bars plus one column per name in FEATURE_COLUMNS.
    Rows near the start of each symbol's history have NaN features (not enough
    past data yet); drop them with df.dropna(subset=FEATURE_COLUMNS).
    """
    df = bars.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    parts = [_symbol_features(g) for _, g in df.groupby("symbol", sort=False)]
    df = df.join(pd.concat(parts))

    hour = df["timestamp"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    market = df.loc[df["symbol"] == market_symbol, ["timestamp", "ret_24", "sma_ratio_168"]]
    if market.empty:
        raise ValueError(f"Market symbol {market_symbol} missing from bars")
    market = market.rename(columns={"ret_24": "mkt_ret_24", "sma_ratio_168": "mkt_sma_ratio_168"})
    df = df.merge(market, on="timestamp", how="left")

    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def add_target(df, horizon, hurdle):
    """future_ret = return over the next `horizon` bars (this IS future data —
    it's the answer we're trying to predict, so it's only used for training/scoring).
    target = 1 if future_ret beats the `hurdle` (round-trip cost)."""
    df = df.copy()
    future_close = df.groupby("symbol")["close"].shift(-horizon)
    df["future_ret"] = future_close / df["close"] - 1
    df["target"] = np.where(df["future_ret"].isna(), np.nan, (df["future_ret"] > hurdle).astype(float))
    return df
