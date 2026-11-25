"""Train the neural network with honest walk-forward testing.

    python -m trader.train

1. Walk-forward: train on the past, predict the next block of time, slide
   forward, repeat. Every prediction is made by a model that never saw that
   period. These "out-of-sample" predictions are saved for the backtester.
2. The same is done for plain logistic regression (your original one-neuron
   model), so you can see whether the bigger network actually helps.
3. A final model is trained on all history and saved for the live bot.
"""
import argparse

import numpy as np
import pandas as pd

from trader import config, nn
from trader.data import load_bars
from trader.features import FEATURE_COLUMNS, add_target, make_features, regularize


def prepare(u, bars=None):
    """Bars -> features + target, ready for training."""
    bars = load_bars(u) if bars is None else bars
    if u["asset_class"] == "crypto":
        bars = regularize(bars)
    df = make_features(bars, u["market_symbol"])
    df = add_target(df, config.HORIZON_BARS, config.round_trip_cost(u))
    return df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


def auc(y, p):
    """Probability a random 'up' example scores higher than a random 'down' one.
    0.5 = no skill, 1.0 = perfect."""
    y = np.asarray(y)
    ranks = pd.Series(p).rank().to_numpy()
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def log_loss(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit(train_df, hidden, seed=config.SEED, verbose=False):
    """Fit scaler + network. The last 15% of the training period is held out
    for early stopping."""
    times = np.sort(train_df["timestamp"].unique())
    cut = times[int(len(times) * 0.85)]
    fit_part = train_df[train_df["timestamp"] < cut]
    val_part = train_df[train_df["timestamp"] >= cut]

    scaler = nn.Standardizer().fit(fit_part[FEATURE_COLUMNS].to_numpy())
    X, y = scaler.transform(fit_part[FEATURE_COLUMNS].to_numpy()), fit_part["target"].to_numpy()
    Xv, yv = scaler.transform(val_part[FEATURE_COLUMNS].to_numpy()), val_part["target"].to_numpy()

    model = nn.MLP(len(FEATURE_COLUMNS), hidden, seed=seed)
    nn.train(model, X, y, Xv, yv, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
             lr=config.LEARNING_RATE, l2=config.L2, patience=config.PATIENCE, seed=seed,
             verbose=verbose)
    return model, scaler


def walk_forward(df, hidden, folds=config.WALK_FORWARD_FOLDS, start=config.WALK_FORWARD_START,
                 horizon=config.HORIZON_BARS):
    """Returns (out-of-sample predictions, per-fold metrics)."""
    times = np.sort(df["timestamp"].unique())
    edges = np.linspace(int(len(times) * start), len(times), folds + 1).astype(int)
    preds, rows = [], []
    for k in range(folds):
        # Purge: the last `horizon` bars before the test block have targets that
        # depend on prices INSIDE the test block. Training on them would leak.
        train_end = times[max(edges[k] - horizon, 1)]
        test_times = times[edges[k]:edges[k + 1]]
        train_df = df[(df["timestamp"] < train_end) & df["target"].notna()]
        test_df = df[df["timestamp"].isin(test_times)].copy()

        model, scaler = fit(train_df, hidden)
        test_df["p_up"] = model.predict_proba(scaler.transform(test_df[FEATURE_COLUMNS].to_numpy()))
        test_df["fold"] = k
        preds.append(test_df)

        scored = test_df[test_df["target"].notna()]
        y, p = scored["target"].to_numpy(), scored["p_up"].to_numpy()
        base = y.mean()
        rows.append({
            "fold": k,
            "test_start": str(pd.Timestamp(test_times[0]).date()),
            "test_end": str(pd.Timestamp(test_times[-1]).date()),
            "base_rate": base,
            "accuracy": ((p > 0.5) == y).mean(),
            "always_no_acc": 1 - base,
            "auc": auc(y, p),
            "logloss": log_loss(y, p),
            "logloss_base": log_loss(y, np.full_like(p, train_df["target"].mean())),
        })
    return pd.concat(preds, ignore_index=True), pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default=config.UNIVERSE)
    parser.add_argument("--skip-walk-forward", action="store_true")
    args = parser.parse_args()

    u = config.universe(args.universe)
    df = prepare(u)
    print(f"{len(df)} rows, {df['symbol'].nunique()} symbols, "
          f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    print(f"Target: next {config.HORIZON_BARS} bars return > {config.round_trip_cost(u):.2%} "
          f"(round-trip cost). Base rate {df['target'].mean():.3f}")
    config.REPORT_DIR.mkdir(exist_ok=True)
    config.MODEL_DIR.mkdir(exist_ok=True)

    if not args.skip_walk_forward:
        pd.set_option("display.width", 140)
        for name, hidden in [("logistic", []), ("mlp", config.HIDDEN_LAYERS)]:
            preds, metrics = walk_forward(df, hidden)
            preds[["timestamp", "symbol", "close", "vol_24", "future_ret", "target", "p_up", "fold"]] \
                .to_csv(config.REPORT_DIR / f"{u['name']}_oos_{name}.csv", index=False)
            print(f"\nWalk-forward: {name} {hidden}")
            print(metrics.round(4).to_string(index=False))
            print(f"Mean AUC {metrics['auc'].mean():.4f}  (0.5 = coin flip)   "
                  f"log-loss {metrics['logloss'].mean():.4f} vs base {metrics['logloss_base'].mean():.4f}")

    print("\nTraining final model on all history...")
    model, scaler = fit(df[df["target"].notna()], config.HIDDEN_LAYERS)
    path = config.MODEL_DIR / f"{u['name']}.npz"
    nn.save(path, model, scaler, {
        "universe": u["name"], "features": FEATURE_COLUMNS, "horizon": config.HORIZON_BARS,
        "hidden": config.HIDDEN_LAYERS, "trained_until": str(df["timestamp"].max()),
    })
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
