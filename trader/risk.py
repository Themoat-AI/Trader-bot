"""Risk management: how big each trade is, and when to stop trading.

The model decides direction. This module decides whether we're *allowed* to
act on it, and with how much money.
"""
from dataclasses import dataclass, field

from trader import config


@dataclass
class RiskLimits:
    position_pct: float = config.POSITION_PCT
    target_vol: float = config.TARGET_VOL
    max_positions: int = config.MAX_POSITIONS
    max_daily_loss: float = config.MAX_DAILY_LOSS
    max_drawdown: float = config.MAX_DRAWDOWN


def position_size(equity, cash, vol, limits=RiskLimits()):
    """Dollars to put into a new position.

    Starts at position_pct of equity, then shrinks when the asset is more
    volatile than target_vol, so a wild coin gets a smaller bet than a calm one.
    Never more than the available cash.
    """
    scale = 1.0 if not vol or vol <= 0 else min(1.0, limits.target_vol / vol)
    return max(0.0, min(equity * limits.position_pct * scale, cash))


@dataclass
class RiskState:
    """Tracks equity over time and says whether trading is allowed."""
    limits: RiskLimits = field(default_factory=RiskLimits)
    day: object = None
    day_start_equity: float = 0.0
    peak_equity: float = 0.0
    killed: bool = False

    def update(self, equity, day):
        if day != self.day:
            self.day, self.day_start_equity = day, equity
        self.peak_equity = max(self.peak_equity, equity)
        if self.peak_equity and equity / self.peak_equity - 1 <= -self.limits.max_drawdown:
            self.killed = True

    def daily_loss_hit(self, equity):
        return self.day_start_equity > 0 and equity / self.day_start_equity - 1 <= -self.limits.max_daily_loss

    def can_open(self, equity, open_positions):
        """(allowed, reason) for opening a NEW position."""
        if self.killed:
            return False, "max drawdown kill switch"
        if self.daily_loss_hit(equity):
            return False, "daily loss limit"
        if open_positions >= self.limits.max_positions:
            return False, "max positions"
        return True, "ok"
