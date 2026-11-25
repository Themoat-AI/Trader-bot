"""The trading rule — shared by the backtester and the live bot.

Because both call decide(), a strategy that works in the backtest is exactly
the strategy that runs live.
"""
from dataclasses import dataclass

from trader import config


@dataclass
class Params:
    enter_threshold: float = config.ENTER_THRESHOLD
    exit_threshold: float = config.EXIT_THRESHOLD
    min_hold_bars: int = config.MIN_HOLD_BARS
    max_hold_bars: int = config.MAX_HOLD_BARS
    stop_loss: float = config.STOP_LOSS


def decide(p_up, holding, bars_held=0, entry_price=None, price=None, params=Params()):
    """Return (action, reason). action is "BUY", "SELL" or "HOLD".

    Risk limits (position count, daily loss, drawdown) are checked separately in
    risk.py — this function only answers "does the model want in or out?".
    """
    if holding:
        if entry_price and price and price / entry_price - 1 <= -params.stop_loss:
            return "SELL", "stop loss"
        if bars_held >= params.max_hold_bars:
            return "SELL", "max hold time"
        if bars_held >= params.min_hold_bars and p_up < params.exit_threshold:
            return "SELL", "model turned negative"
        return "HOLD", "in position"
    if p_up > params.enter_threshold:
        return "BUY", "model confident"
    return "HOLD", "no signal"
