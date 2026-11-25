from trader.risk import RiskLimits, RiskState, position_size
from trader.strategy import Params, decide

P = Params(enter_threshold=0.6, exit_threshold=0.45, min_hold_bars=6, max_hold_bars=48, stop_loss=0.05)


def test_entry_rules():
    assert decide(0.7, False, params=P)[0] == "BUY"
    assert decide(0.55, False, params=P)[0] == "HOLD"


def test_exit_rules():
    assert decide(0.3, True, bars_held=2, entry_price=100, price=100, params=P)[0] == "HOLD"   # too fresh
    assert decide(0.3, True, bars_held=6, entry_price=100, price=100, params=P)[0] == "SELL"
    assert decide(0.5, True, bars_held=10, entry_price=100, price=100, params=P)[0] == "HOLD"  # in the gap
    assert decide(0.9, True, bars_held=48, entry_price=100, price=100, params=P) == ("SELL", "max hold time")
    assert decide(0.9, True, bars_held=1, entry_price=100, price=94, params=P) == ("SELL", "stop loss")


def test_position_size():
    limits = RiskLimits(position_pct=0.10, target_vol=0.01)
    assert position_size(100_000, 100_000, 0.005, limits) == 10_000    # calm: full size
    assert position_size(100_000, 100_000, 0.02, limits) == 5_000      # 2x volatile: half size
    assert position_size(100_000, 3_000, 0.005, limits) == 3_000       # never more than cash


def test_daily_loss_and_max_positions():
    r = RiskState(RiskLimits(max_positions=2, max_daily_loss=0.02, max_drawdown=0.15))
    r.update(100_000, "d1")
    assert r.can_open(100_000, 0) == (True, "ok")
    assert r.can_open(100_000, 2)[1] == "max positions"
    assert r.can_open(97_500, 0)[1] == "daily loss limit"
    r.update(97_500, "d2")                       # new day resets the daily limit
    assert r.can_open(97_500, 0) == (True, "ok")


def test_drawdown_kill_switch():
    r = RiskState(RiskLimits(max_drawdown=0.15))
    r.update(100_000, "d1")
    r.update(120_000, "d2")
    r.update(101_000, "d3")
    assert r.killed
    assert r.can_open(101_000, 0)[1] == "max drawdown kill switch"
