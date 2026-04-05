"""
Apache Cassandra integration for prediction logging.

Connects to a local Cassandra instance (default: 127.0.0.1:9042).
If Cassandra is not running the module degrades gracefully — all public
functions return safe fallback values so the rest of the app is unaffected.

Keyspace  : bone_cancer_db
Table     : prediction_logs

Quick-start (Docker)
--------------------
    docker run -d --name cassandra -p 9042:9042 cassandra:4.1

Install driver
--------------
    pip install cassandra-driver
"""

import uuid
from datetime import datetime

# ── Optional import ───────────────────────────────────────────────────────────
try:
    from cassandra.cluster import Cluster, NoHostAvailable
    from cassandra.policies import RoundRobinPolicy
    _DRIVER_OK = True
except ImportError:
    _DRIVER_OK = False

# ── Constants ─────────────────────────────────────────────────────────────────
CASSANDRA_HOSTS = ["127.0.0.1"]
CASSANDRA_PORT  = 9042
KEYSPACE        = "bone_cancer_db"
TABLE           = "prediction_logs"

# Module-level cached session (avoids reconnecting on every call)
_session = None
_cluster = None


# ── Connection ────────────────────────────────────────────────────────────────

def connect_cassandra():
    """
    Connect to Cassandra, create keyspace + table if absent.

    Returns
    -------
    cassandra.cluster.Session on success, None on failure.
    """
    global _session, _cluster

    if not _DRIVER_OK:
        return None                     # driver not installed
    if _session is not None:
        return _session                 # reuse existing session

    try:
        _cluster = Cluster(
            CASSANDRA_HOSTS,
            port=CASSANDRA_PORT,
            load_balancing_policy=RoundRobinPolicy(),
            connect_timeout=4,
            control_connection_timeout=4,
        )
        session = _cluster.connect()
        _create_keyspace(session)
        session.set_keyspace(KEYSPACE)
        _create_table(session)
        _session = session
        return _session
    except Exception:
        _session = None
        return None


def disconnect():
    """Close the Cassandra connection."""
    global _session, _cluster
    try:
        if _cluster:
            _cluster.shutdown()
    except Exception:
        pass
    _session = None
    _cluster  = None


def is_connected() -> bool:
    """Return True if a live Cassandra session exists."""
    return connect_cassandra() is not None


# ── DDL helpers ───────────────────────────────────────────────────────────────

def _create_keyspace(session):
    session.execute(f"""
        CREATE KEYSPACE IF NOT EXISTS {KEYSPACE}
        WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': '1'}}
        AND durable_writes = true
    """)


def _create_table(session):
    """
    prediction_logs schema
    ----------------------
    id            UUID   PRIMARY KEY
    image_name    TEXT
    prediction    TEXT   (normal | benign | malignant)
    confidence    FLOAT
    normal_prob   FLOAT
    benign_prob   FLOAT
    malignant_prob FLOAT
    timestamp     TEXT
    """
    session.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            id              UUID PRIMARY KEY,
            image_name      TEXT,
            prediction      TEXT,
            confidence      FLOAT,
            normal_prob     FLOAT,
            benign_prob     FLOAT,
            malignant_prob  FLOAT,
            timestamp       TEXT
        )
    """)


# ── DML helpers ───────────────────────────────────────────────────────────────

def insert_prediction(
    image_name:    str,
    prediction:    str,
    confidence:    float,
    probabilities: dict,
    timestamp:     str = None,
) -> bool:
    """
    Insert one prediction record into prediction_logs.

    Parameters
    ----------
    image_name    : filename of the X-ray (e.g. "xray1.jpg")
    prediction    : class label  ("normal" | "benign" | "malignant")
    confidence    : max-class probability × 100  (e.g. 87.3)
    probabilities : dict mapping class name → probability (0–100)
    timestamp     : ISO-format string; auto-generated if None

    Returns
    -------
    True on success, False on any error.
    """
    session = connect_cassandra()
    if session is None:
        return False

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        session.execute(
            f"""
            INSERT INTO {TABLE}
                (id, image_name, prediction, confidence,
                 normal_prob, benign_prob, malignant_prob, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                uuid.uuid4(),
                image_name,
                prediction,
                float(confidence),
                float(probabilities.get("normal",    0)),
                float(probabilities.get("benign",    0)),
                float(probabilities.get("malignant", 0)),
                timestamp,
            ),
        )
        return True
    except Exception:
        return False


def get_recent_predictions(limit: int = 100) -> list:
    """
    Fetch up to `limit` rows from prediction_logs.

    Returns list of dicts (empty list if Cassandra unavailable).
    Note: Cassandra does not guarantee ORDER BY without clustering columns,
    so results are sorted by timestamp in Python.
    """
    session = connect_cassandra()
    if session is None:
        return []

    try:
        rows = session.execute(
            f"""
            SELECT id, image_name, prediction, confidence,
                   normal_prob, benign_prob, malignant_prob, timestamp
            FROM {TABLE}
            LIMIT {limit}
            """
        )
        records = [
            {
                "id":            str(row.id),
                "image_name":    row.image_name,
                "prediction":    row.prediction,
                "confidence":    round(float(row.confidence), 2),
                "normal_prob":   round(float(row.normal_prob),    2),
                "benign_prob":   round(float(row.benign_prob),    2),
                "malignant_prob":round(float(row.malignant_prob), 2),
                "timestamp":     row.timestamp,
            }
            for row in rows
        ]
        # Sort newest first
        records.sort(key=lambda r: r["timestamp"], reverse=True)
        return records
    except Exception:
        return []


def get_prediction_stats() -> dict:
    """
    Return aggregate counts per class from the table.
    Returns empty dict if Cassandra is unavailable.
    """
    session = connect_cassandra()
    if session is None:
        return {}

    try:
        rows = session.execute(f"SELECT prediction FROM {TABLE}")
        counts = {"normal": 0, "benign": 0, "malignant": 0, "total": 0}
        for row in rows:
            p = (row.prediction or "").lower()
            if p in counts:
                counts[p] += 1
            counts["total"] += 1
        return counts
    except Exception:
        return {}
