from datetime import datetime, timezone

import numpy as np

from tests.conftest import make_bars
from trader.bot import run_cycle
from trader.journal import Journal
from trader.risk import RiskState
from trader.strategy import Params

U = {"name": "test", "asset_class": "crypto", "symbols": ["BTC/USD", "ETH/USD"], "market_symbol": "BTC/USD"}
NOW = datetime(2025, 1, 20, 0, 30, tzinfo=timezone.utc)


class FakeBroker:
    dry_run = False

    def __init__(self, positions=None):
        self.pos = positions or {}
        self.bought, self.sold = [], []

    def market_open(self):
        return True

    def account(self):
        return {"equity": 100_000.0, "cash": 100_000.0, "last_equity": 100_000.0}

    def positions(self):
        return dict(self.pos)

    def buy(self, symbol, dollars):
        self.bought.append((symbol, dollars))
        return "id-buy"

    def sell(self, symbol):
        self.sold.append(symbol)
        return "id-sell"


class ConstModel:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.full(len(X), self.p)


class Identity:
    def transform(self, X):
        return X


def fetch(u, start, end=None):
    return make_bars(n=600)


def test_buys_when_confident(tmp_path):
    broker = FakeBroker()
    journal = Journal(tmp_path / "j.db", "test")
    acts = run_cycle(U, ConstModel(0.9), Identity(), broker, journal, RiskState(), fetch=fetch, now=NOW)
    assert {a[1] for a in acts} == {"BUY"}
    assert len(broker.bought) == 2
    assert journal.entry_time("BTC/USD") is not None


def test_sells_when_model_turns_negative(tmp_path):
    broker = FakeBroker({"BTC/USD": {"qty": 0.1, "avg_entry_price": 100.0, "market_value": 10.0}})
    journal = Journal(tmp_path / "j.db", "test")
    run_cycle(U, ConstModel(0.1), Identity(), broker, journal, RiskState(), Params(stop_loss=10),
              fetch=fetch, now=NOW)
    assert broker.sold == ["BTC/USD"] and broker.bought == []


def test_uses_only_finished_bars(tmp_path):
    from trader.bot import latest_features
    latest, _ = latest_features(U, fetch, now=NOW)
    assert (latest["timestamp"] <= NOW.replace(minute=0) - np.timedelta64(1, "h")).all()


def test_skips_bar_already_handled(tmp_path):
    from trader.bot import last_finished_bar
    journal = Journal(tmp_path / "j.db", "test")
    bar = last_finished_bar(NOW)
    assert str(bar) == "2025-01-19 23:00:00+00:00"
    assert not journal.has_bar(bar)
    run_cycle(U, ConstModel(0.1), Identity(), FakeBroker(), journal, RiskState(), fetch=fetch, now=NOW)
    assert journal.has_bar(bar)
