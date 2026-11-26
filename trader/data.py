"""Download and cache historical bars from Alpaca.

    python -m trader.data                 # download the configured universe
    python -m trader.data --universe stocks
"""
import argparse
from datetime import datetime, timedelta, timezone

import pandas as pd

from trader import config

BAR_COLUMNS = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]


def fetch_bars(u, start, end=None):
    """Hourly bars for every symbol in universe `u`, as one tidy DataFrame."""
    from alpaca.data.timeframe import TimeFrame

    if u["asset_class"] == "crypto":
        from alpaca.data.historical import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest

        client = CryptoHistoricalDataClient()
        request = CryptoBarsRequest(symbol_or_symbols=u["symbols"], timeframe=TimeFrame.Hour,
                                    start=start, end=end)
        raw = client.get_crypto_bars(request).df
    else:
        from alpaca.data.enums import DataFeed
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest

        client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
        request = StockBarsRequest(symbol_or_symbols=u["symbols"], timeframe=TimeFrame.Hour,
                                   start=start, end=end, feed=DataFeed.IEX)  # IEX feed is free
        raw = client.get_stock_bars(request).df

    df = raw.reset_index()[BAR_COLUMNS]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if u["asset_class"] == "stock":
        df = regular_session(df)
    missing = sorted(set(u["symbols"]) - set(df["symbol"]))
    if missing:
        print(f"Warning: no data returned for {missing}")
    return df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def regular_session(df):
    """Keep the 7 hourly bars that start 9:00-15:00 New York time (the regular
    session). Pre- and after-market bars are thin and noisy on the free feed."""
    hour = df["timestamp"].dt.tz_convert("America/New_York").dt.hour
    weekday = df["timestamp"].dt.tz_convert("America/New_York").dt.weekday
    return df[(hour >= 9) & (hour <= 15) & (weekday < 5)]


def drop_unfinished(df, now=None, bar=timedelta(hours=1)):
    """Remove bars whose hour hasn't finished yet — they'd change after we read them."""
    now = now or datetime.now(timezone.utc)
    return df[df["timestamp"] + bar <= now]


def bars_path(u):
    return config.DATA_DIR / f"{u['name']}_bars.csv"


def save_bars(df, u):
    config.DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(bars_path(u), index=False)


def load_bars(u):
    path = bars_path(u)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m trader.data --universe {u['name']}")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default=config.UNIVERSE)
    parser.add_argument("--days", type=int, default=config.HISTORY_DAYS)
    args = parser.parse_args()

    u = config.universe(args.universe)
    start = datetime.now(timezone.utc) - timedelta(days=args.days)
    print(f"Downloading {args.days} days of hourly bars for {u['symbols']}...")
    df = drop_unfinished(fetch_bars(u, start))
    save_bars(df, u)
    print(df.groupby("symbol").size().rename("bars").to_string())
    print(f"Saved {len(df)} rows to {bars_path(u)}")


if __name__ == "__main__":
    main()
