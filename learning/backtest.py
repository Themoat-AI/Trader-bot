import numpy as np
import pandas as pd

# ---------- Load data and model ----------
df = pd.read_csv("btc_features.csv")
model = np.load("model.npz")
w, b, mean, std = model["w"], model["b"], model["mean"], model["std"]

features = ["ret_1", "ret_5", "sma_ratio"]
market_symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
threshold = 0.60
fee = 0.0025

markets = market_symbols if "symbol" in df.columns else ["BTC/USD"]
for symbol in markets:
    market_df = df[df["symbol"] == symbol].copy() if "symbol" in df.columns else df.copy()
    market_df = market_df.sort_values("timestamp") if "timestamp" in market_df.columns else market_df
    if len(market_df) < 10:
        print(symbol, "- not enough test data")
        continue

    split = int(len(market_df) * 0.8)
    test = market_df.iloc[split:].copy()
    test["future_ret"] = pd.to_numeric(test["future_ret"], errors="coerce").fillna(0.0)

    X = (test[features].values - mean) / std
    test["p_up"] = 1 / (1 + np.exp(-(X @ w + b)))
    test["position"] = (test["p_up"] > threshold).astype(int)

    trades = test["position"].diff().abs().fillna(test["position"].iloc[0])
    test["strat_ret"] = test["position"] * test["future_ret"] - trades * fee

    strategy_total = np.prod(1.0 + test["strat_ret"].to_numpy(dtype=float)) - 1.0
    buy_hold_total = np.prod(1.0 + test["future_ret"].to_numpy(dtype=float)) - 1.0

    print(symbol)
    print("Prediction range:", round(test["p_up"].min(), 4), "to", round(test["p_up"].max(), 4))
    print("Number of trades:", int(trades.sum()))
    print("Time in market:", round(test["position"].mean() * 100, 1), "%")
    print("Strategy return:", round(strategy_total * 100, 2), "%")
    print("Buy & hold return:", round(buy_hold_total * 100, 2), "%")