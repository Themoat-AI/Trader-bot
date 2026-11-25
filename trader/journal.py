"""SQLite trade journal: every decision, order and equity snapshot the bot makes.

The dashboard reads from here, and the bot reads entry times back after a
restart so it knows how long it has held each position.
"""
import sqlite3
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    time TEXT, universe TEXT, symbol TEXT, bar_time TEXT, price REAL, p_up REAL,
    holding INTEGER, action TEXT, reason TEXT, dry_run INTEGER);
CREATE TABLE IF NOT EXISTS orders (
    time TEXT, universe TEXT, symbol TEXT, side TEXT, dollars REAL, price REAL,
    order_id TEXT, reason TEXT, dry_run INTEGER);
CREATE TABLE IF NOT EXISTS equity (
    time TEXT, universe TEXT, equity REAL, cash REAL, open_positions INTEGER);
CREATE TABLE IF NOT EXISTS entries (
    universe TEXT, symbol TEXT, entry_time TEXT, PRIMARY KEY (universe, symbol));
"""


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Journal:
    def __init__(self, path, universe):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)
        self.universe = universe

    def decision(self, symbol, bar_time, price, p_up, holding, action, reason, dry_run):
        self.db.execute("INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (now_iso(), self.universe, symbol, str(bar_time), price, p_up,
                         int(holding), action, reason, int(dry_run)))
        self.db.commit()

    def order(self, symbol, side, dollars, price, order_id, reason, dry_run):
        self.db.execute("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?)",
                        (now_iso(), self.universe, symbol, side, dollars, price, order_id,
                         reason, int(dry_run)))
        self.db.commit()

    def snapshot(self, equity, cash, open_positions):
        self.db.execute("INSERT INTO equity VALUES (?,?,?,?,?)",
                        (now_iso(), self.universe, equity, cash, open_positions))
        self.db.commit()

    def set_entry(self, symbol, entry_time):
        self.db.execute("INSERT OR REPLACE INTO entries VALUES (?,?,?)",
                        (self.universe, symbol, str(entry_time)))
        self.db.commit()

    def clear_entry(self, symbol):
        self.db.execute("DELETE FROM entries WHERE universe=? AND symbol=?", (self.universe, symbol))
        self.db.commit()

    def entry_time(self, symbol):
        row = self.db.execute("SELECT entry_time FROM entries WHERE universe=? AND symbol=?",
                              (self.universe, symbol)).fetchone()
        return row[0] if row else None

    def peak_equity(self):
        row = self.db.execute("SELECT MAX(equity) FROM equity WHERE universe=?", (self.universe,)).fetchone()
        return row[0] or 0.0
