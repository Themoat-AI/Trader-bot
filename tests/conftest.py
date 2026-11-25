import numpy as np
import pandas as pd
import pytest


def make_bars(symbols=("BTC/USD", "ETH/USD"), n=400, seed=0, start="2025-01-01"):
    rng = np.random.default_rng(seed)
    times = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    frames = []
    for i, s in enumerate(symbols):
        close = 100 * (i + 1) * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        frames.append(pd.DataFrame({"symbol": s, "timestamp": times, "open": close, "high": close * 1.01,
                                    "low": close * 0.99, "close": close,
                                    "volume": rng.uniform(1, 10, n)}))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def bars():
    return make_bars()
