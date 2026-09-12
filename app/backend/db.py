"""
RESONA Database Layer
SQLite-backed persistent storage for events, alerts, reviews, replay, model versions.
"""
import sqlite3
import json
import os
from datetime import datetime
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), 'resona.db')

SCHEMA = """
CREATE TABLE IF NOT EXISTS inference_events (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    machine_id TEXT NOT NULL,
    prediction INTEGER,
    confidence REAL,
    mahalanobis_distance REAL,
    ood_threshold REAL,
    is_ood INTEGER,
    ood_calibrated INTEGER,
    decision TEXT,
    preprocessing_latency_ms REAL,
    inference_latency_ms REAL,
    total_latency_ms REAL,
    model_version TEXT,
    waveform_json TEXT,
    spectrogram_json TEXT,
    sample_rate INTEGER,
    duration_sec REAL
);

CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY,
    machine_id TEXT NOT NULL,
    zone TEXT,
    severity TEXT,
    decision TEXT,
    first_detected TEXT NOT NULL,
    last_detected TEXT NOT NULL,
    event_count INTEGER DEFAULT 1,
    status TEXT DEFAULT 'active',
    acknowledged INTEGER DEFAULT 0,
    acknowledged_at TEXT,
    last_event_id TEXT
);

CREATE TABLE IF NOT EXISTS reviews (
    id TEXT PRIMARY KEY,
    machine_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    ood_score REAL,
    model_version TEXT,
    created_at TEXT NOT NULL,
    status TEXT DEFAULT 'pending',
    human_label TEXT,
    operator_notes TEXT,
    resolved_at TEXT,
    FOREIGN KEY(event_id) REFERENCES inference_events(id)
);

CREATE TABLE IF NOT EXISTS replay_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id TEXT NOT NULL,
    label INTEGER NOT NULL,
    human_label TEXT,
    review_id TEXT,
    npy_path TEXT,
    added_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_versions (
    version TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    parent_version TEXT,
    status TEXT DEFAULT 'candidate',
    accuracy REAL,
    forgetting REAL,
    pth_path TEXT,
    ood_artifact_path TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS learning_jobs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    status TEXT DEFAULT 'queued',
    trigger_review_id TEXT,
    candidate_version TEXT,
    result_notes TEXT
);
"""

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)
    print(f"[DB] Initialized at {DB_PATH}")

def insert_event(event: dict):
    sql = """INSERT OR REPLACE INTO inference_events
        (id, timestamp, machine_id, prediction, confidence, mahalanobis_distance,
         ood_threshold, is_ood, ood_calibrated, decision, preprocessing_latency_ms,
         inference_latency_ms, total_latency_ms, model_version,
         waveform_json, spectrogram_json, sample_rate, duration_sec)
        VALUES (:id, :timestamp, :machine_id, :prediction, :confidence, :mahalanobis_distance,
         :ood_threshold, :is_ood, :ood_calibrated, :decision, :preprocessing_latency_ms,
         :inference_latency_ms, :total_latency_ms, :model_version,
         :waveform_json, :spectrogram_json, :sample_rate, :duration_sec)"""
    with get_conn() as conn:
        conn.execute(sql, event)

def upsert_alert(machine_id, zone, decision, event_id, severity='medium'):
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT id, event_count FROM alerts WHERE machine_id=? AND status='active' AND decision=?",
            (machine_id, decision)
        ).fetchone()
        if existing:
            conn.execute("""UPDATE alerts SET last_detected=?, event_count=event_count+1, last_event_id=?
                           WHERE id=?""", (now, event_id, existing['id']))
            return existing['id'], False  # (alert_id, is_new)
        else:
            import uuid
            alert_id = str(uuid.uuid4())
            conn.execute("""INSERT INTO alerts (id, machine_id, zone, severity, decision, first_detected,
                           last_detected, event_count, status, last_event_id)
                           VALUES (?, ?, ?, ?, ?, ?, ?, 1, 'active', ?)""",
                        (alert_id, machine_id, zone, severity, decision, now, now, event_id))
            return alert_id, True

def resolve_alert(machine_id, decision):
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        conn.execute("""UPDATE alerts SET status='resolved', last_detected=?
                       WHERE machine_id=? AND decision=? AND status='active'""",
                    (now, machine_id, decision))

def acknowledge_alert(alert_id):
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        conn.execute("UPDATE alerts SET acknowledged=1, acknowledged_at=? WHERE id=?",
                    (now, alert_id))

def get_active_alerts():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM alerts WHERE status='active' ORDER BY first_detected DESC").fetchall()
        return [dict(r) for r in rows]

def get_events(limit=50):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM inference_events ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

def insert_review(review: dict):
    sql = """INSERT INTO reviews (id, machine_id, event_id, ood_score, model_version,
             created_at, status) VALUES (:id, :machine_id, :event_id, :ood_score,
             :model_version, :created_at, :status)"""
    with get_conn() as conn:
        conn.execute(sql, review)

def get_pending_reviews():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM reviews WHERE status='pending' ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

def get_all_reviews(limit=20):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM reviews ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

def get_review(review_id):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM reviews WHERE id=?", (review_id,)).fetchone()
        return dict(row) if row else None

def update_review(review_id, human_label, notes=''):
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        conn.execute("""UPDATE reviews SET status='resolved', human_label=?, operator_notes=?,
                       resolved_at=? WHERE id=?""", (human_label, notes, now, review_id))

def add_replay_sample(machine_id, label, human_label, review_id, npy_path=None):
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        conn.execute("""INSERT INTO replay_samples (machine_id, label, human_label, review_id, npy_path, added_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (machine_id, label, human_label, review_id, npy_path, now))

def get_replay_count():
    with get_conn() as conn:
        row = conn.execute("SELECT COUNT(*) as cnt FROM replay_samples").fetchone()
        return row['cnt']

def insert_learning_job(job: dict):
    sql = """INSERT INTO learning_jobs (id, created_at, status, trigger_review_id)
             VALUES (:id, :created_at, :status, :trigger_review_id)"""
    with get_conn() as conn:
        conn.execute(sql, job)

def update_learning_job(job_id, **kwargs):
    if not kwargs:
        return
    sets = ', '.join(f"{k}=?" for k in kwargs)
    vals = list(kwargs.values()) + [job_id]
    with get_conn() as conn:
        conn.execute(f"UPDATE learning_jobs SET {sets} WHERE id=?", vals)

def get_latest_learning_job():
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM learning_jobs ORDER BY created_at DESC LIMIT 1").fetchone()
        return dict(row) if row else None

def register_model_version(version, pth_path, ood_path, parent_version=None, notes=''):
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        conn.execute("""INSERT OR REPLACE INTO model_versions (version, created_at, parent_version,
                       status, pth_path, ood_artifact_path, notes)
                       VALUES (?, ?, ?, 'candidate', ?, ?, ?)""",
                    (version, now, parent_version, pth_path, ood_path, notes))

def promote_model_version(version, accuracy, forgetting):
    with get_conn() as conn:
        conn.execute("UPDATE model_versions SET status='active', accuracy=?, forgetting=? WHERE version=?",
                    (accuracy, forgetting, version))
        conn.execute("UPDATE model_versions SET status='retired' WHERE status='active' AND version!=?",
                    (version,))

def get_active_model_version():
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM model_versions WHERE status='active' ORDER BY created_at DESC LIMIT 1").fetchone()
        return dict(row) if row else None


if __name__ == '__main__':
    init_db()
    print("DB initialized successfully.")
