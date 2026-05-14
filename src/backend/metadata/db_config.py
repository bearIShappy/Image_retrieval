"""
Central configuration for Qdrant + SQLite storage.

Load order (highest priority first):
  1. OS environment variables (set by shell)
  2. Project-root ``.env`` file (loaded via python-dotenv)
  3. Hard-coded defaults below

All paths, collection names, and DB constants live here so every
module imports from a single source of truth.
"""

import os
from enum import Enum

# ── Load .env from project root ───────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _ROOT = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    _env_path = os.path.join(_ROOT, ".env")
    if os.path.isfile(_env_path):
        load_dotenv(_env_path, override=False)
except ImportError:
    pass   # python-dotenv not installed → rely on OS env vars only


class DatasetType(str, Enum):
    """Enum for dataset source labels."""

    TRAINING = "TRAINING"
    SUPPORT  = "SUPPORT"
    TEST     = "TEST"


# ── Project root ──────────────────────────────────────────────────────────────
# db_config.py lives at src/backend/metadata/ — 3 levels below the repo root
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
)

# ── Qdrant ────────────────────────────────────────────────────────────────────
# QDRANT_HOST: set to "localhost" for both direct and qdrant.exe execution.
# When falsy the code falls back to an on-disk local Qdrant instance.
QDRANT_HOST       = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT       = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_PATH       = os.path.join(PROJECT_ROOT, "qdrant_data")   # embedded fallback
QDRANT_COLLECTION = "clip_embeddings"
EMBEDDING_DIM     = 512   # CLIP ViT-B/32 projection dimension

# ── SQLite metadata store ─────────────────────────────────────────────────────
SQLITE_DB_PATH = os.path.join(PROJECT_ROOT, "src", "metadata.db")

# ── Legacy pickle (for migration) ────────────────────────────────────────────
LEGACY_EMBEDDINGS_PKL = os.path.join(PROJECT_ROOT, "embeddings.pkl")
LEGACY_TEST_PKL       = os.path.join(PROJECT_ROOT, "embeddings_test.pkl")

# ── Dataset directories ───────────────────────────────────────────────────────
DATASET_DIR         = os.path.join(PROJECT_ROOT, "dataset")
CLEANED_DATASET_DIR = os.path.join(PROJECT_ROOT, "cleaned_dataset")
TEST_DATASET_DIR    = os.path.join(PROJECT_ROOT, "test_dataset")

# ── Fixed allowed support classes ────────────────────────────────────────────
ALLOWED_SUPPORT_CLASSES = [
    "CFF with and without load",
    "heavy drop",
    "para motor",
    "static line jump"
]

# ── Image status states ───────────────────────────────────────────────────────
# PENDING : uploaded but not yet trained/indexed
# ACTIVE  : successfully trained and visible in dataset
# FAILED  : training failed, not visible
VALID_IMAGE_STATUSES = ["PENDING", "ACTIVE", "FAILED"]


# ── Convenience helper ────────────────────────────────────────────────────────

def describe() -> dict:
    """
    Return a human-readable summary of active config.

    Returns:
        dict: Config snapshot safe to log/print.
    """
    return {
        "qdrant":        f"{QDRANT_HOST}:{QDRANT_PORT}",
        "collection":    QDRANT_COLLECTION,
        "embedding_dim": EMBEDDING_DIM,
        "sqlite_db":     SQLITE_DB_PATH,
    }
