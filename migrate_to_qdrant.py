"""
One-shot migration script: moves all data from embeddings.pkl
into Qdrant + SQLite.

Run once:
    python migrate_to_qdrant.py

This will:
  1. Read embeddings.pkl and embeddings_test.pkl
  2. Upsert all vectors into Qdrant (local on-disk)
  3. Populate the SQLite dataset_images manifest
  4. Print stats

After migration your pkl files are NOT deleted (backup).
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.backend.metadata.vector_store import VectorStore, migrate_pkl_to_qdrant
from src.backend.metadata.metadata_db import MetadataDB
from src.backend.metadata.db_config import LEGACY_EMBEDDINGS_PKL, LEGACY_TEST_PKL
from src.backend.metadata.db_config import QDRANT_PATH  
from PIL import Image

lock_file = os.path.join(QDRANT_PATH, ".lock")
if os.path.exists(lock_file):
    print(f"  Removing stale lock file: {lock_file}")
    os.remove(lock_file)

def main():
    print("=" * 60)
    print("  Qdrant + SQLite Migration")
    print("=" * 60)

    store = VectorStore()
    db = MetadataDB()

    # Migrate main embeddings
    if os.path.isfile(LEGACY_EMBEDDINGS_PKL):
        n = migrate_pkl_to_qdrant(LEGACY_EMBEDDINGS_PKL, store, source="dataset")
        print(f"  Main dataset: {n} points migrated")
    else:
        print(f"  No main pickle found at {LEGACY_EMBEDDINGS_PKL}")

    # Migrate test embeddings
    if os.path.isfile(LEGACY_TEST_PKL):
        n = migrate_pkl_to_qdrant(LEGACY_TEST_PKL, store, source="test")
        print(f"  Test dataset: {n} points migrated")
    else:
        print(f"  No test pickle found at {LEGACY_TEST_PKL}")

    # Populate SQLite manifest from Qdrant payload
    print("\nPopulating SQLite manifest...")
    globals_dict = store.get_all_global_vectors()
    entries = []
    for path in globals_dict:
        parent = os.path.basename(os.path.dirname(path))
        cls = parent if parent else "unknown"
        source = "test" if "test_dataset" in path else "dataset"
        w, h = None, None
        try:
            img = Image.open(path)
            w, h = img.size
        except Exception:
            pass
        entries.append({
            "path": path, "class": cls, "source": source,
            "width": w, "height": h, "n_regions": 0,
        })
    db.upsert_images_batch(entries)

    # Print stats
    print("\n" + "=" * 60)
    print(f"  Qdrant total points:  {store.count()}")
    print(f"  Qdrant dataset pts:   {store.count('dataset')}")
    print(f"  Qdrant test pts:      {store.count('test')}")
    print(f"  SQLite stats:         {db.stats()}")
    print("=" * 60)
    print("\nMigration complete. Your .pkl files have been kept as backups.")


if __name__ == "__main__":
    main()
