"""Place one small PAPER round trip to prove orders work end to end.

    python -m trader.test_order               # $100 of the first coin the bot doesn't hold
    python -m trader.test_order --dollars 50

Buys, waits for the fill, then sells the whole position.
"""
import argparse
import time

from trader import config
from trader.broker import AlpacaBroker


def wait_for_fill(client, order_id, timeout=60):
    for _ in range(timeout):
        order = client.get_order_by_id(order_id)
        status = getattr(order.status, "value", order.status)
        if status == "filled":
            return order
        time.sleep(1)
    raise SystemExit(f"Order {order_id} not filled after {timeout}s (status {order.status})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dollars", type=float, default=100.0)
    args = parser.parse_args()

    u = config.universe("crypto")
    broker = AlpacaBroker(u, dry_run=False)
    held = broker.positions()
    free = [s for s in u["symbols"] if s not in held]
    if not free:
        raise SystemExit("The bot holds every coin right now; not touching its positions.")
    symbol = free[0]   # never trade a coin the bot is holding

    before = broker.account()
    print(f"Paper account equity before: ${before['equity']:,.2f}")
    print(f"Buying ${args.dollars:.2f} of {symbol}...")
    buy = wait_for_fill(broker.client, broker.buy(symbol, args.dollars))
    print(f"  filled {float(buy.filled_qty):.8f} @ ${float(buy.filled_avg_price):,.2f}")

    time.sleep(5)
    print(f"Selling {symbol}...")
    sell = wait_for_fill(broker.client, broker.sell(symbol))
    print(f"  filled {float(sell.filled_qty):.8f} @ ${float(sell.filled_avg_price):,.2f}")

    after = broker.account()
    print(f"Paper account equity after:  ${after['equity']:,.2f} "
          f"(change ${after['equity'] - before['equity']:+,.2f}, mostly fees)")
    print("Orders work. Check them in the Alpaca paper dashboard under Orders.")


if __name__ == "__main__":
    main()
