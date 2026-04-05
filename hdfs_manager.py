"""
Simulated HDFS (Hadoop Distributed File System) using local folder structure.

Mimics the HDFS block storage concept for academic Big Data integration.
Each directory represents a logical HDFS namespace path.

HDFS Layout
-----------
hdfs/
  raw_images/       ← original uploads (as received from user)
  processed_images/ ← resized 224×224 images fed to the model
  predictions/      ← JSON prediction records (one file per inference)
  logs/             ← system-level event logs
"""

import os
import json
import shutil
from datetime import datetime
from PIL import Image


# ── HDFS root (lives inside the project directory) ────────────────────────────
HDFS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hdfs")

HDFS_DIRS = {
    "raw":         os.path.join(HDFS_ROOT, "raw_images"),
    "processed":   os.path.join(HDFS_ROOT, "processed_images"),
    "predictions": os.path.join(HDFS_ROOT, "predictions"),
    "logs":        os.path.join(HDFS_ROOT, "logs"),
}


# ── Initialisation ────────────────────────────────────────────────────────────

def init_hdfs() -> str:
    """Create all HDFS namespace directories if they do not exist."""
    for path in HDFS_DIRS.values():
        os.makedirs(path, exist_ok=True)
    return HDFS_ROOT


# ── Write operations ──────────────────────────────────────────────────────────

def save_raw_image(image_bytes: bytes, filename: str) -> str:
    """
    Write the raw uploaded image bytes to hdfs/raw_images/.

    Parameters
    ----------
    image_bytes : raw file bytes (from st.file_uploader read)
    filename    : original filename (e.g. "xray_001.jpg")

    Returns
    -------
    Absolute path of saved file.
    """
    init_hdfs()
    dest = os.path.join(HDFS_DIRS["raw"], filename)
    with open(dest, "wb") as fh:
        fh.write(image_bytes)
    _log_event("WRITE", f"raw_images/{filename}", len(image_bytes))
    return dest


def save_processed_image(pil_image: Image.Image, filename: str) -> str:
    """
    Resize image to 224×224 and write to hdfs/processed_images/.

    Returns absolute path of saved file.
    """
    init_hdfs()
    processed = pil_image.resize((224, 224))
    dest = os.path.join(HDFS_DIRS["processed"], filename)
    processed.save(dest)
    size = os.path.getsize(dest)
    _log_event("WRITE", f"processed_images/{filename}", size)
    return dest


def save_prediction_json(
    image_name:    str,
    prediction:    str,
    confidence:    float,
    probabilities: dict,
    timestamp:     str = None,
) -> str:
    """
    Persist a prediction record as JSON to hdfs/predictions/.

    The file is named prediction_NNNN.json where NNNN auto-increments.

    Returns absolute path of saved file.
    """
    init_hdfs()
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Auto-increment ID based on existing files
    existing = [f for f in os.listdir(HDFS_DIRS["predictions"]) if f.endswith(".json")]
    pred_id  = len(existing) + 1

    record = {
        "id":            pred_id,
        "image":         image_name,
        "prediction":    prediction,
        "confidence":    round(float(confidence), 2),
        "probabilities": {k: round(float(v), 2) for k, v in probabilities.items()},
        "timestamp":     timestamp,
    }

    dest = os.path.join(HDFS_DIRS["predictions"], f"prediction_{pred_id:04d}.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)

    _log_event("WRITE", f"predictions/prediction_{pred_id:04d}.json", os.path.getsize(dest))
    return dest


# ── Read operations ───────────────────────────────────────────────────────────

def list_predictions(limit: int = 100) -> list:
    """
    Load and return prediction records from hdfs/predictions/, newest first.

    Returns list of dicts.
    """
    init_hdfs()
    pred_dir = HDFS_DIRS["predictions"]
    files = sorted(
        [f for f in os.listdir(pred_dir) if f.endswith(".json")],
        reverse=True,
    )[:limit]

    records = []
    for fname in files:
        try:
            with open(os.path.join(pred_dir, fname), encoding="utf-8") as fh:
                records.append(json.load(fh))
        except Exception:
            pass
    return records


def get_hdfs_stats() -> dict:
    """
    Return file count and total size (MB) for each HDFS namespace.

    Example return value::

        {
          "raw":         {"count": 12, "size_mb": 4.7},
          "processed":   {"count": 12, "size_mb": 1.2},
          "predictions": {"count": 12, "size_mb": 0.01},
          "logs":        {"count":  1, "size_mb": 0.002},
        }
    """
    init_hdfs()
    stats = {}
    for name, path in HDFS_DIRS.items():
        try:
            files      = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
            total_size = sum(os.path.getsize(os.path.join(path, f)) for f in files)
            stats[name] = {
                "count":   len(files),
                "size_mb": round(total_size / (1024 * 1024), 3),
            }
        except Exception:
            stats[name] = {"count": 0, "size_mb": 0.0}
    return stats


def clear_hdfs():
    """Remove all files from every HDFS namespace (useful for testing)."""
    for path in HDFS_DIRS.values():
        if os.path.isdir(path):
            shutil.rmtree(path)
    init_hdfs()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _log_event(operation: str, path: str, size_bytes: int = 0):
    """Append one line to hdfs/logs/hdfs.log."""
    try:
        log_file = os.path.join(HDFS_DIRS["logs"], "hdfs.log")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a", encoding="utf-8") as fh:
            fh.write(f"[{ts}] {operation:6s}  /{path}  ({size_bytes} bytes)\n")
    except Exception:
        pass
