"""Print a Markdown summary of the bot's journal.

    python -m trader.report

GitHub Actions shows this on each run's page, so you can follow the bot from
the repo's Actions tab without opening the database.
"""
import argparse
import sqlite3

import pandas as pd

from trader import config


def table(df):
    if df.empty:
        return "_none_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def summary(universe):
    if not config.JOURNAL_DB.exists():
        return "No journal yet: the bot hasn't completed a cycle.\n"
    db = sqlite3.connect(config.JOURNAL_DB)
    q = lambda sql: pd.read_sql(sql, db, params=(universe,))  # noqa: E731
    equity = q("SELECT time, equity, cash, open_positions FROM equity WHERE universe=? ORDER BY time")
    decisions = q("SELECT time, symbol, price, p_up, action, reason, dry_run FROM decisions "
                  "WHERE universe=? ORDER BY time DESC LIMIT 50")
    orders = q("SELECT time, symbol, side, dollars, price, reason, dry_run FROM orders "
               "WHERE universe=? ORDER BY time DESC LIMIT 10")

    out = [f"## Trader bot: {universe}\n"]
    if not equity.empty:
        first, last = equity["equity"].iloc[0], equity["equity"].iloc[-1]
        peak = equity["equity"].max()
        out.append(f"**Equity:** ${last:,.2f} ({last / first - 1:+.2%} since the bot started, "
                   f"{last / peak - 1:+.2%} from peak)  \n"
                   f"**Cash:** ${equity['cash'].iloc[-1]:,.2f}  \n"
                   f"**Open positions:** {int(equity['open_positions'].iloc[-1])}  \n"
                   f"**Cycles run:** {len(equity)}\n")
    if not decisions.empty:
        latest = decisions[decisions["time"] == decisions["time"].iloc[0]].copy()
        mode = "DRY RUN (no orders)" if latest["dry_run"].iloc[0] else "LIVE paper trading"
        out.append(f"### Latest cycle ({latest['time'].iloc[0]} UTC) · {mode}\n")
        latest["p_up"] = latest["p_up"].round(3)
        latest["price"] = latest["price"].round(2)
        out.append(table(latest[["symbol", "price", "p_up", "action", "reason"]]))
    out.append("### Recent orders\n")
    if not orders.empty:
        orders["dollars"] = orders["dollars"].round(2)
        orders["price"] = orders["price"].round(2)
    out.append(table(orders))
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default=config.UNIVERSE)
    print(summary(parser.parse_args().universe))


if __name__ == "__main__":
    main()
