"""
SQLite-backed storage for store-manager overrides of system recommendations.

Every override is logged with BOTH the system's recommendation and the
manager's final number. This is what eventually gets folded back into
the next training run as a supervised signal (see fold_into_training.py).
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "feedback.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS overrides (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    store_id TEXT NOT NULL,
    sku_id TEXT NOT NULL,
    forecast_date TEXT NOT NULL,
    system_recommendation REAL NOT NULL,
    manager_override REAL NOT NULL,
    override_reason TEXT,
    folded_into_training INTEGER DEFAULT 0
);
"""


@contextmanager
def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute(SCHEMA)
        conn.commit()


def log_override(
    store_id: str,
    sku_id: str,
    forecast_date: str,
    system_recommendation: float,
    manager_override: float,
    override_reason: str = "",
) -> int:
    init_db()
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO overrides
               (timestamp, store_id, sku_id, forecast_date, system_recommendation, manager_override, override_reason)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now(timezone.utc).isoformat(), store_id, sku_id, forecast_date,
             system_recommendation, manager_override, override_reason),
        )
        conn.commit()
        return cur.lastrowid


def get_unfolded_overrides():
    """Overrides not yet incorporated into a training run."""
    init_db()
    with get_conn() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM overrides WHERE folded_into_training = 0").fetchall()
        return [dict(r) for r in rows]


def mark_folded(ids: list[int]):
    if not ids:
        return
    with get_conn() as conn:
        conn.executemany("UPDATE overrides SET folded_into_training = 1 WHERE id = ?", [(i,) for i in ids])
        conn.commit()


if __name__ == "__main__":
    init_db()
    rid = log_override(
        store_id="STORE_1", sku_id="SKU_0002", forecast_date="2024-11-12",
        system_recommendation=84.0, manager_override=120.0,
        override_reason="Local cricket match expected to spike footfall",
    )
    print(f"Logged override id={rid}")
    print("Unfolded overrides:", get_unfolded_overrides())
