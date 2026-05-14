"""
Dataset indexer: compute CLIP embeddings for all dataset images.

Now backed by Qdrant (vectors) + SQLite (metadata) instead of pickle.

Backward-compatible: still exposes build_index / load_index / save_index
but they now read/write through the DB layer. Legacy pickle helpers are
retained for migration.

Supports two index formats (in-memory dict, for retriever compat):
  - Legacy:   {image_path: tensor(1,D)}
  - Extended: {image_path: {"global_embedding": tensor, "regions": [region_meta]}}
"""

import os
import pickle
import torch
from PIL import Image
from typing import Dict, Optional, Any, Union
from src.backend.retrieval.clip_model import CLIPEncoder

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def _is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


# ── Format helpers ───────────────────────────────────────────────────────────

def get_global_embedding(entry: Any) -> torch.Tensor:
    """
    Extract the global embedding from an index entry.

    Works with both legacy (tensor) and extended (dict) formats.

    Args:
        entry: Either a tensor(1,D) or a dict with "global_embedding" key.

    Returns:
        tensor(1, D).
    """
    if isinstance(entry, dict):
        return entry["global_embedding"]
    return entry  # legacy: entry is a tensor


def get_regions(entry: Any) -> list:
    """
    Extract region metadata list from an index entry.

    Returns empty list for legacy format.
    """
    if isinstance(entry, dict) and "regions" in entry:
        return entry["regions"]
    return []


def is_extended_index(index: dict) -> bool:
    """Check if this index uses extended format (has regions)."""
    if not index:
        return False
    first = next(iter(index.values()))
    return isinstance(first, dict) and "global_embedding" in first


# ── Build index (Qdrant + SQLite backed) ─────────────────────────────────────

def build_index(
    clip: CLIPEncoder,
    dataset_dir: str,
    cache_path: str = "embeddings.pkl",
    use_multiscale: bool = True,
    use_regions: bool = False,
    source: str = "dataset",
    vector_store=None,
    metadata_db=None,
) -> Dict[str, Any]:
    """
    Walk dataset_dir, compute CLIP embeddings for every image.

    If vector_store and metadata_db are provided, writes to Qdrant + SQLite.
    Otherwise falls back to pickle (legacy behavior).

    Returns:
        Legacy:   {image_path: tensor(1, D)}
        Extended: {image_path: {"global_embedding": tensor, "regions": [...]}}
    """
    index: Dict[str, Any] = {}
    image_paths = []

    for root, _, files in os.walk(dataset_dir):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            if _is_image(fpath):
                image_paths.append(fpath)

    print(f"[indexer] Found {len(image_paths)} images in {dataset_dir}")
    if use_regions:
        print("[indexer] Extended format enabled (global + regions)")

    db_entries = []  # for batch SQLite upsert

    for i, path in enumerate(image_paths, 1):
        try:
            img = Image.open(path).convert("RGB")
            w, h = img.size

            if use_multiscale:
                global_emb = clip.encode_image_multiscale(img, scales=[224, 384])
            else:
                global_emb = clip.encode_image_pil(img)

            regions = []
            if use_regions:
                from src.backend.region_aware.detector import extract_region_metadata_pil
                regions = extract_region_metadata_pil(
                    clip, img,
                    scales=[0.3, 0.5, 0.7],
                    min_px=32,
                    min_area_ratio=0.02,
                    max_area_ratio=0.80,
                    batch_size=16,
                )

            # Build in-memory index (for retriever compat)
            if use_regions:
                index[path] = {
                    "global_embedding": global_emb,
                    "regions": regions,
                }
            else:
                index[path] = global_emb

            # Infer class from parent directory
            parent = os.path.basename(os.path.dirname(path))
            cls = parent if parent else "unknown"

            # Write to Qdrant
            if vector_store is not None:
                vector_store.upsert_image(
                    path=path,
                    global_embedding=global_emb,
                    regions=regions if use_regions else None,
                    cls=cls,
                    source=source,
                )

            # Collect for SQLite batch
            if metadata_db is not None:
                db_entries.append({
                    "path": path, "class": cls, "source": source,
                    "width": w, "height": h,
                    "n_regions": len(regions),
                })

            if i % 10 == 0 or i == len(image_paths):
                print(f"  [{i}/{len(image_paths)}] indexed")
        except Exception as e:
            print(f"  [{i}/{len(image_paths)}] SKIP {os.path.basename(path)}: {e}")

    # Batch write to SQLite
    if metadata_db is not None and db_entries:
        metadata_db.upsert_images_batch(db_entries)

    # Also save legacy pickle for backward compat
    if cache_path:
        save_index(index, cache_path)

    return index


def save_index(index: Dict[str, Any], cache_path: str = "embeddings.pkl") -> None:
    """Save embedding index to pickle (legacy compat)."""
    with open(cache_path, "wb") as f:
        pickle.dump(index, f)
    fmt = "extended" if is_extended_index(index) else "legacy"
    print(f"[indexer] Saved {len(index)} embeddings -> {cache_path} ({fmt})")


def load_index(cache_path: str = "embeddings.pkl") -> Optional[Dict[str, Any]]:
    """Load embedding index from pickle if exists (legacy compat)."""
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, "rb") as f:
        index = pickle.load(f)
    fmt = "extended" if is_extended_index(index) else "legacy"
    print(f"[indexer] Loaded {len(index)} embeddings from {cache_path} ({fmt})")
    return index


def load_index_from_qdrant(vector_store) -> Dict[str, Any]:
    """
    Reconstruct the in-memory index dict from Qdrant.
    Returns extended format: {path: {"global_embedding": ..., "regions": [...]}}
    """
    # Get all global vectors
    globals_dict = vector_store.get_all_global_vectors()

    # For each image, also get its regions
    index = {}
    for path, global_emb in globals_dict.items():
        entry = vector_store.get_image_entry(path)
        if entry:
            index[path] = entry
        else:
            index[path] = {"global_embedding": global_emb, "regions": []}

    print(f"[indexer] Loaded {len(index)} images from Qdrant")
    return index
