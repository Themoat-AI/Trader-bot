"""Thin wrapper around Alpaca's paper-trading API.

Everything the bot needs from the broker goes through these few methods, so
tests can swap in a fake broker and the rest of the code doesn't care.
"""
from trader import config


def broker_symbol(symbol):
    """Alpaca reports crypto positions without the slash: BTC/USD -> BTCUSD."""
    return symbol.replace("/", "")


class AlpacaBroker:
    def __init__(self, u, dry_run=True):
        from alpaca.trading.client import TradingClient

        if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
            raise SystemExit("Alpaca keys not found. Put ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
        self.u, self.dry_run = u, dry_run
        self.client = TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, paper=True)  # paper only
        self._by_broker_symbol = {broker_symbol(s): s for s in u["symbols"]}

    def account(self):
        a = self.client.get_account()
        return {"equity": float(a.equity), "cash": float(a.cash), "last_equity": float(a.last_equity)}

    def positions(self):
        """Open positions in this universe: {symbol: {qty, avg_entry_price, market_value}}."""
        out = {}
        for p in self.client.get_all_positions():
            symbol = self._by_broker_symbol.get(p.symbol)
            if symbol:
                out[symbol] = {"qty": float(p.qty), "avg_entry_price": float(p.avg_entry_price),
                               "market_value": float(p.market_value)}
        return out

    def market_open(self):
        if self.u["asset_class"] == "crypto":
            return True
        return self.client.get_clock().is_open

    def buy(self, symbol, dollars):
        if self.dry_run:
            return "dry-run"
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        tif = TimeInForce.GTC if self.u["asset_class"] == "crypto" else TimeInForce.DAY
        order = self.client.submit_order(MarketOrderRequest(
            symbol=symbol, notional=round(dollars, 2), side=OrderSide.BUY, time_in_force=tif))
        return str(order.id)

    def sell(self, symbol):
        if self.dry_run:
            return "dry-run"
        order = self.client.close_position(broker_symbol(symbol))
        return str(getattr(order, "id", "closed"))
