# ---------- The neuron ----------
import numpy as np
import pandas as pd


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def compute_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))

eps = 1e-9
lr = 0.1

df = pd.read_csv("btc_features.csv")
feature_columns = ["ret_1", "ret_5", "sma_ratio"]

train_frames = []
for _, market_df in df.groupby("symbol") if "symbol" in df.columns else [(None, df)]:
    market_df = market_df.sort_values("timestamp") if "timestamp" in market_df.columns else market_df
    split = int(len(market_df) * 0.8)
    train_frames.append(market_df.iloc[:split])

train_df = pd.concat(train_frames, ignore_index=True)
X_train = train_df[feature_columns].to_numpy(dtype=float)
y_train = train_df["target"].to_numpy(dtype=float)

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
std = np.where(std == 0, 1.0, std)

X_train = (X_train - mean) / std

w = np.zeros(X_train.shape[1])
b = 0.0
n = len(y_train)

for step in range(1000):
   
    p = sigmoid(X_train @ w + b)
    loss = compute_loss(y_train, p)

    
    dw = X_train.T @ (p - y_train) / n
    db = np.mean(p - y_train)

    
    w = w - lr * dw
    b = b - lr * db

    if step % 100 == 0:
        print(f"step {step}  loss {loss:.5f}")

np.savez("model.npz", w=w, b=b, mean=mean, std=std)
print("Saved model.npz")