from datetime import datetime, timedelta
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

# ---------- Part A: download about 2 years of hourly BTC prices ----------
client = CryptoHistoricalDataClient()  # crypto data doesn't need keys

MARKET_SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD"]

request = CryptoBarsRequest(
    symbol_or_symbols=MARKET_SYMBOLS,
    timeframe=TimeFrame.Hour,
    start=datetime.now() - timedelta(days=730),
)

bars = client.get_crypto_bars(request)
df = bars.df.reset_index()
print("Downloaded rows:", len(df))

# ---------- Part B: features (only use the past) ----------
df = df.sort_values(["symbol", "timestamp"])
df["ret_1"] = df.groupby("symbol")["close"].pct_change()
df["ret_5"] = df.groupby("symbol")["close"].pct_change(5)
sma_20 = df.groupby("symbol")["close"].transform(lambda s: s.rolling(20).mean())
df["sma_ratio"] = df["close"] / sma_20 - 1      # % above/below the average

# ---------- What we're predicting (the future) ----------
df["future_ret"] = df.groupby("symbol")["close"].pct_change(-1).mul(-1)

# Remove rows with missing values BEFORE making the target
df = df.dropna()
df["target"] = (df["future_ret"] > 0).astype(int)  # 1 = price went up next hour

# ---------- Check and save ----------
print(df[["timestamp", "close", "ret_1", "ret_5", "sma_ratio", "target"]].tail())
print("Rows after cleaning:", len(df))
print("Fraction of hours that went up:", round(df["target"].mean(), 3))

df.to_csv("btc_features.csv", index=False)
print("Saved to btc_features.csv")