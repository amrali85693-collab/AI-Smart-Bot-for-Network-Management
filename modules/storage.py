"""SQLite persistence layer for chat history, alerts, metrics, and remediation events."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path(__file__).resolve().parent.parent / "network_bot.db"


def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Establish a connection to the SQLite database with optimized parameters (WAL mode, timeouts)."""
    target_path = db_path or DB_PATH
    conn = sqlite3.connect(target_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        # Enable WAL mode for better concurrency and write resilience
        conn.execute("PRAGMA journal_mode=WAL;")
    except sqlite3.OperationalError:
        # Fallback if journal_mode setting fails in locked envs
        pass
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize SQLite tables if they do not exist."""
    path = db_path or DB_PATH
    with _connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS alert_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_json TEXT NOT NULL,
                explanation TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS remediation_incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_id TEXT NOT NULL UNIQUE,
                alert_id TEXT,
                device TEXT NOT NULL,
                issue_type TEXT NOT NULL,
                state TEXT NOT NULL,
                detected_at TEXT NOT NULL,
                diagnosed_at TEXT,
                suggested_at TEXT,
                approved_at TEXT,
                rejected_at TEXT,
                diagnosis TEXT,
                suggested_action_type TEXT,
                suggested_action_description TEXT,
                suggested_action_severity TEXT,
                suggested_action_impact TEXT,
                suggested_action_rollback TEXT,
                approval_notes TEXT,
                rejection_reason TEXT,
                state_history TEXT,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.commit()


def save_chat_message(
    role: str,
    content: str,
    sources: Optional[List[str]] = None,
    db_path: Optional[Path] = None,
) -> None:
    """Save a user or assistant chat message to the history."""
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO chat_messages (role, content, sources, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                role,
                content,
                json.dumps(sources or []),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()


def save_alert(
    alert: Dict[str, Any],
    explanation: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> None:
    """Save a network threshold alert event alongside its AI explanation."""
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO alert_events (alert_json, explanation, created_at)
            VALUES (?, ?, ?)
            """,
            (
                json.dumps(alert),
                explanation,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()


def get_history(limit: int = 50, db_path: Optional[Path] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Retrieve historical chat messages and alerts."""
    init_db(db_path)
    with _connect(db_path) as conn:
        chat_rows = conn.execute(
            """
            SELECT role, content, sources, created_at
            FROM chat_messages
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        alert_rows = conn.execute(
            """
            SELECT alert_json, explanation, created_at
            FROM alert_events
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return {
        "chat": [dict(row) for row in chat_rows],
        "alerts": [dict(row) for row in alert_rows],
    }


def clear_chat_history(db_path: Optional[Path] = None) -> None:
    """Delete all saved chatbot history from the database."""
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM chat_messages")
        conn.commit()


def save_evaluation_event(
    event_type: str,
    query: str,
    retrieval_score: Optional[float] = None,
    retrieval_hit: bool = False,
    latency_ms: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
    db_path: Optional[Path] = None,
) -> None:
    """
    Save evaluation metrics for latency measurement and RAG analysis.
    Useful for scoring performance metrics.
    """
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluation_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                query TEXT NOT NULL,
                retrieval_score REAL,
                retrieval_hit INTEGER,
                latency_ms REAL,
                metadata TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO evaluation_events (event_type, query, retrieval_score, retrieval_hit, latency_ms, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_type,
                query,
                retrieval_score,
                1 if retrieval_hit else 0,
                latency_ms,
                json.dumps(metadata or {}),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()


def get_evaluation_metrics(event_type: Optional[str] = None, db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Retrieve evaluation metrics for RAG hits and system latency."""
    init_db(db_path)
    with _connect(db_path) as conn:
        if event_type:
            rows = conn.execute(
                """
                SELECT retrieval_hit, retrieval_score, latency_ms, created_at
                FROM evaluation_events
                WHERE event_type = ?
                ORDER BY id DESC
                """,
                (event_type,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT retrieval_hit, retrieval_score, latency_ms, created_at
                FROM evaluation_events
                ORDER BY id DESC
                """
            ).fetchall()

    if not rows:
        return {"total_events": 0, "hit_rate": 0.0, "avg_latency_ms": None, "avg_score": None}

    hits = sum(1 for row in rows if row["retrieval_hit"])
    scores = [row["retrieval_score"] for row in rows if row["retrieval_score"] is not None]
    latencies = [row["latency_ms"] for row in rows if row["latency_ms"] is not None]

    return {
        "total_events": len(rows),
        "hit_rate": hits / len(rows),
        "avg_score": sum(scores) / len(scores) if scores else None,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else None,
    }


def save_remediation_incident(incident_data: Dict[str, Any], db_path: Optional[Path] = None) -> None:
    """Save or update a remediation incident in the database."""
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO remediation_incidents (
                incident_id, alert_id, device, issue_type, state,
                detected_at, diagnosed_at, suggested_at, approved_at, rejected_at,
                diagnosis, suggested_action_type, suggested_action_description,
                suggested_action_severity, suggested_action_impact, suggested_action_rollback,
                approval_notes, rejection_reason, state_history, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                incident_data.get("incident_id"),
                incident_data.get("alert_id"),
                incident_data.get("device"),
                incident_data.get("issue_type"),
                incident_data.get("state"),
                incident_data.get("detected_at"),
                incident_data.get("diagnosed_at"),
                incident_data.get("suggested_at"),
                incident_data.get("approved_at"),
                incident_data.get("rejected_at"),
                incident_data.get("diagnosis"),
                incident_data.get("suggested_action_type"),
                incident_data.get("suggested_action_description"),
                incident_data.get("suggested_action_severity"),
                incident_data.get("suggested_action_impact"),
                incident_data.get("suggested_action_rollback"),
                incident_data.get("approval_notes"),
                incident_data.get("rejection_reason"),
                json.dumps(incident_data.get("state_history", [])),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()


def get_remediation_incidents(
    limit: int = 50,
    state: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Retrieve historical remediation incidents filtered optionally by state."""
    init_db(db_path)
    with _connect(db_path) as conn:
        if state:
            rows = conn.execute(
                """
                SELECT * FROM remediation_incidents
                WHERE state = ?
                ORDER BY detected_at DESC
                LIMIT ?
                """,
                (state, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM remediation_incidents
                ORDER BY detected_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    incidents: List[Dict[str, Any]] = []
    for row in rows:
        incident = dict(row)
        if incident.get("state_history"):
            try:
                incident["state_history"] = json.loads(incident["state_history"])
            except (json.JSONDecodeError, TypeError):
                incident["state_history"] = []
        incidents.append(incident)

    return incidents
