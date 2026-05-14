import os
import time
import json
import glob
import torch
import traceback
import threading
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from src.backend.retrieval.clip_model import CLIPEncoder
from src.backend.indexing.indexer import build_index, load_index, load_index_from_qdrant
from src.backend.retrieval.retriever import build_query_embedding, retrieve_with_prototypes, retrieve_with_qdrant
from src.backend.training.finetune import build_prototypes
from src.backend.metadata.vector_store import VectorStore, migrate_pkl_to_qdrant
from src.backend.metadata.metadata_db import MetadataDB
from src.backend.metadata.db_config import (
    QDRANT_PATH, SQLITE_DB_PATH, LEGACY_EMBEDDINGS_PKL, LEGACY_TEST_PKL,
    ALLOWED_SUPPORT_CLASSES,
)

app = FastAPI()

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Training state (for async finetune) ────────────────────────────────────
_training_state = {
    "status": "idle",   # idle | running | success | error
    "message": "",
    "progress": "",
    "result": None,
}


# Mount static dirs only if they exist — missing dirs crash StaticFiles on startup
_dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
if os.path.exists(_dataset_dir):
    app.mount("/dataset", StaticFiles(directory=_dataset_dir), name="dataset")

_cleaned_dir = os.path.join(PROJECT_ROOT, "cleaned_dataset")
if os.path.exists(_cleaned_dir):
    app.mount("/cleaned_dataset", StaticFiles(directory=_cleaned_dir), name="cleaned_dataset")
else:
    print(f"[WARNING] cleaned_dataset/ not found at {_cleaned_dir} — /cleaned_dataset static route disabled")

test_dir_path = os.path.join(PROJECT_ROOT, "test_dataset")
os.makedirs(test_dir_path, exist_ok=True)
app.mount("/test_dataset", StaticFiles(directory=test_dir_path), name="test_dataset")

clip = None
index = None
prototypes = None
vector_store = None
metadata_db = None


def _index_support_dataset(clip_enc, vs, mdb):
    """
    Index cleaned_dataset/ into Qdrant as source="SUPPORT".
    Each class subfolder becomes the class label.
    """
    support_dir = os.path.join(PROJECT_ROOT, "cleaned_dataset")
    if not os.path.exists(support_dir):
        print("[startup] cleaned_dataset/ not found, skipping support indexing")
        return {}

    existing = vs.count(source="SUPPORT")
    if existing > 0:
        print(f"[startup] Support dataset already indexed ({existing} points), skipping")
        return {}

    print("[startup] Indexing cleaned_dataset as support dataset...")
    support_index = build_index(
        clip_enc, support_dir,
        cache_path="",  # no pickle needed
        use_regions=False,
        source="SUPPORT",
        vector_store=vs,
        metadata_db=mdb,
    )
    print(f"[startup] Support dataset: {len(support_index)} images indexed")
    return support_index


def _index_main_dataset(clip_enc, vs, mdb):
    """Index dataset/ into Qdrant as source="TRAINING"."""
    dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
    if not os.path.exists(dataset_dir):
        print("[startup] dataset/ not found, skipping")
        return {}

    existing = vs.count(source="TRAINING")
    if existing > 0:
        print(f"[startup] Main dataset already indexed ({existing} points), skipping")
        return {}

    print("[startup] Indexing dataset/...")
    main_index = build_index(
        clip_enc, dataset_dir,
        cache_path="",
        use_regions=True,
        source="TRAINING",
        vector_store=vs,
        metadata_db=mdb,
    )
    print(f"[startup] Main dataset: {len(main_index)} images indexed")
    return main_index


def _index_test_dataset(clip_enc, vs, mdb):
    """Index test_dataset/ into Qdrant as source="TEST"."""
    test_dir = os.path.join(PROJECT_ROOT, "test_dataset")
    if not os.path.exists(test_dir):
        return {}

    # Count actual image files
    img_exts = {'.png', '.jpg', '.jpeg', '.webp'}
    img_files = [f for f in os.listdir(test_dir)
                 if os.path.splitext(f)[1].lower() in img_exts]
    if not img_files:
        return {}

    existing = vs.count(source="TEST")
    if existing >= len(img_files):
        print(f"[startup] Test dataset already indexed ({existing} points), skipping")
        return {}

    print(f"[startup] Indexing {len(img_files)} test images (global only, no regions for speed)...")
    test_index = build_index(
        clip_enc, test_dir,
        cache_path="",
        use_regions=False,   # regions on CPU are very slow; skip at startup
        source="TEST",
        vector_store=vs,
        metadata_db=mdb,
    )
    print(f"[startup] Test dataset: {len(test_index)} images indexed")
    return test_index


import glob
from src.backend.training.finetune import load_finetuned_weights, finetune_clip

def get_latest_model_path():
    models = glob.glob(os.path.join(PROJECT_ROOT, "finetuned_clip*.pt"))
    if not models:
        return None
    latest_path = None
    max_ver = -1
    for m in models:
        base = os.path.basename(m)
        if base == "finetuned_clip.pt":
            ver = 0
        else:
            try:
                ver = int(base.replace("finetuned_clip", "").replace(".pt", ""))
            except ValueError:
                continue
        if ver >= max_ver:
            max_ver = ver
            latest_path = m
    return latest_path, max_ver

@app.on_event("startup")
def load_models():
    global clip, index, prototypes, vector_store, metadata_db
    print("=" * 60)
    print("  CLIP Retrieval Backend — Starting up")
    print("=" * 60)

    # ── 1. Load CLIP ─────────────────────────────────────────────────────
    print("\n[startup] Loading CLIP model...")
    clip = CLIPEncoder(model_path=os.path.join(PROJECT_ROOT, "models", "clip-vit-b-32"))
    
    # Check for latest finetuned model
    latest_model_info = get_latest_model_path()
    if latest_model_info and latest_model_info[0]:
        latest_path, ver = latest_model_info
        print(f"\n[startup] Loading latest fine-tuned weights from {latest_path}...")
        try:
            load_finetuned_weights(clip, latest_path)
            print("[startup] Successfully loaded fine-tuned weights!")
        except Exception as e:
            print(f"[startup] Failed to load fine-tuned weights: {e}")

    # ── 2. Initialize Qdrant + SQLite ────────────────────────────────────
    print("\n[startup] Initializing Qdrant vector store...")
    vector_store = VectorStore()

    print("[startup] Initializing SQLite metadata DB...")
    metadata_db = MetadataDB()

    # ── 3. Auto-migrate from pickle if Qdrant is completely empty ────────
    if vector_store.count() == 0:
        pkl_path = os.path.join(PROJECT_ROOT, "embeddings.pkl")
        if os.path.isfile(pkl_path):
            print("\n[startup] Qdrant is empty — migrating from embeddings.pkl...")
            migrate_pkl_to_qdrant(pkl_path, vector_store, source="TRAINING")

    # ── 4. Index cleaned_dataset as support ──────────────────────────────
    support_index = _index_support_dataset(clip, vector_store, metadata_db)

    # ── 5. Index main dataset ────────────────────────────────────────────
    main_index = _index_main_dataset(clip, vector_store, metadata_db)

    # ── 6. Index test dataset ────────────────────────────────────────────
    test_index = _index_test_dataset(clip, vector_store, metadata_db)

    # ── 7. Load full in-memory index from Qdrant ─────────────────────────
    print("\n[startup] Loading full index from Qdrant...")
    if vector_store.count() > 0:
        index = load_index_from_qdrant(vector_store)
    else:
        index = {}

    # ── 8. Build prototypes from support set ─────────────────────────────
    print("\n[startup] Building class prototypes from cleaned_dataset...")
    prototypes = build_prototypes(clip, os.path.join(PROJECT_ROOT, "cleaned_dataset"))

    # Save prototypes to SQLite
    metadata_db.clear_prototypes()
    for cls_name, proto_emb in prototypes.items():
        metadata_db.save_prototype(cls_name, proto_emb, n_images=0)

    # ── Done ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Qdrant: {vector_store.count()} total points")
    print(f"    - dataset:  {vector_store.count('TRAINING')} points")
    print(f"    - support:  {vector_store.count('SUPPORT')} points")
    print(f"    - test:     {vector_store.count('TEST')} points")
    print(f"  SQLite: {metadata_db.stats()}")
    print(f"  In-memory index: {len(index)} images")
    print(f"  Prototypes: {list(prototypes.keys())}")
    print(f"  Docker needed: NO (Qdrant local + SQLite built-in)")
    print("=" * 60)
    print("  Backend ready at http://localhost:5000")
    print("=" * 60 + "\n")


@app.get("/api/dataset")
async def get_dataset():
    support_dir = os.path.join(PROJECT_ROOT, "cleaned_dataset")
    test_dir = os.path.join(PROJECT_ROOT, "test_dataset")

    # Build set of ACTIVE image paths from SQLite
    # NULL status = legacy record (indexed before status column existed) → treat as ACTIVE
    def _is_active(row):
        s = row.get("status")
        return s is None or s.upper() in ("ACTIVE", "")

    indexed_paths: set = set()
    if metadata_db:
        for row in metadata_db.get_images():
            if _is_active(row):
                p = row["path"]
                indexed_paths.add(os.path.normpath(p))
                indexed_paths.add(p.replace("\\\\", "/").replace("\\", "/"))

    support_classes = []
    if os.path.exists(support_dir):
        for cls_name in sorted(os.listdir(support_dir)):
            # SKIP _pending_uploads and any non-directory
            if cls_name.startswith("_"):
                continue
            cls_path = os.path.join(support_dir, cls_name)
            if not os.path.isdir(cls_path):
                continue
            images = []
            images_detail = []
            for img_name in sorted(os.listdir(cls_path)):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.avif', '.bmp', '.tiff')):
                    continue
                abs_path = os.path.normpath(os.path.join(cls_path, img_name))
                url = f"/cleaned_dataset/{cls_name}/{img_name}"
                images.append(url)
                images_detail.append({
                    "url": url,
                    "filename": img_name,
                    "path": abs_path,
                    "indexed": abs_path in indexed_paths,
                })
            # Only add class if it has active images
            if images:
                support_classes.append({
                    "name": cls_name,
                    "images": images,
                    "images_detail": images_detail,
                })
    else:
        print(f"[api/dataset] WARNING: cleaned_dataset/ not found at {support_dir}")

    test_images = []
    test_detail = []
    if os.path.exists(test_dir):
        for img_name in sorted(os.listdir(test_dir)):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.avif', '.bmp', '.tiff')):
                continue
            url = f"/test_dataset/{img_name}"
            abs_path = os.path.normpath(os.path.join(test_dir, img_name))
            test_images.append(url)
            test_detail.append({
                "url": url,
                "filename": img_name,
                "indexed": abs_path in indexed_paths,
                "path": abs_path,
            })
    main_images = []
    main_detail = []
    main_dir = os.path.join(PROJECT_ROOT, "dataset")
    if os.path.exists(main_dir):
        for img_name in sorted(os.listdir(main_dir)):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.avif', '.bmp', '.tiff')):
                continue
            url = f"/dataset/{img_name}"
            abs_path = os.path.normpath(os.path.join(main_dir, img_name))
            main_images.append(url)
            main_detail.append({
                "url": url,
                "filename": img_name,
                "indexed": abs_path in indexed_paths,
                "path": abs_path,
            })

    return {
        "support": support_classes,
        "test": test_images,
        "test_detail": test_detail,
        "main": main_images,
        "main_detail": main_detail,
        "stats": {
            "support_classes": len(support_classes),
            "support_images": sum(len(c["images"]) for c in support_classes),
            "support_images_pending": metadata_db.pending_image_count() if metadata_db else 0,
            "test_images": len(test_images),
            "qdrant_support": vector_store.count("SUPPORT") if vector_store else 0,
            "qdrant_support_active": metadata_db.image_count("SUPPORT", status="ACTIVE") if metadata_db else 0,
            "qdrant_test": vector_store.count("TEST") if vector_store else 0,
            "qdrant_total": vector_store.count() if vector_store else 0,
        }
    }


def get_unique_filepath(directory: str, filename: str) -> tuple[str, str]:
    """Returns (absolute_path, unique_filename) ensuring no overwrite."""
    base, ext = os.path.splitext(filename)
    unique_name = filename
    path = os.path.join(directory, unique_name)
    counter = 1
    while os.path.exists(path):
        unique_name = f"{base}_{counter}{ext}"
        path = os.path.join(directory, unique_name)
        counter += 1
    return path, unique_name

@app.post("/api/upload-support")
async def upload_support(
    files: list[UploadFile] = File(...),
    classes: list[str] = Form(...),
):
    """
    Upload support images with assigned classes.
    
    STRICT ENFORCEMENT:
    - Each file must have a corresponding class
    - Classes must be from ALLOWED_SUPPORT_CLASSES only
    - Images are saved as PENDING (not indexed until training succeeds)
    - Upload metadata (image_path, upload_date, image_name, image_class)
      is persisted to SQLite immediately after file save.
    """
    from datetime import datetime, timezone

    if not metadata_db:
        return {"status": "error", "message": "Database not initialized"}
    
    # Validate: same number of files and classes
    if len(files) != len(classes):
        return {
            "status": "error",
            "message": f"Mismatch: {len(files)} files but {len(classes)} classes"
        }
    
    # Wait! No staging directory needed anymore. Save directly to class folder.
    support_dir = os.path.join(PROJECT_ROOT, "cleaned_dataset")
    
    success_count = 0
    failed = []
    
    for img, class_name in zip(files, classes):
        # VALIDATION: Class must be in allowed list
        if class_name not in ALLOWED_SUPPORT_CLASSES:
            failed.append({
                "filename": img.filename,
                "class": class_name,
                "reason": f"Unknown class. Allowed: {ALLOWED_SUPPORT_CLASSES}"
            })
            continue
        
        # Create class directory if it doesn't exist
        class_dir = os.path.join(support_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Ensure unique filename
        file_path, img.filename = get_unique_filepath(class_dir, img.filename)
        
        try:
            # Save permanently to class folder
            with open(file_path, "wb") as f:
                f.write(await img.read())
            
            # Capture upload timestamp immediately after file save
            upload_date = datetime.now(timezone.utc).isoformat()

            # Register in database as PENDING (awaiting fine-tuning)
            if metadata_db.add_pending_image(file_path, img.filename, class_name):
                # Also record it as an unindexed image in dataset_images
                # with full upload metadata
                metadata_db.upsert_image(
                    path=file_path,
                    cls=class_name,
                    source="SUPPORT",
                    status="PENDING",
                    image_name=img.filename,
                    upload_date=upload_date,
                )
                success_count += 1
                print(
                    f"[upload-support] Saved: {img.filename} -> {class_name} "
                    f"(upload_date={upload_date})"
                )
            else:
                failed.append({
                    "filename": img.filename,
                    "class": class_name,
                    "reason": "Failed to register in database"
                })
                # Rollback
                if os.path.exists(file_path):
                    os.remove(file_path)
        except Exception as e:
            failed.append({
                "filename": img.filename,
                "class": class_name,
                "reason": str(e)
            })
            # Rollback
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
    
    # Get updated stats to return to frontend
    stats = {
        "support_images_pending": metadata_db.pending_image_count() if metadata_db else 0,
        "qdrant_support_active": metadata_db.image_count("SUPPORT", status="ACTIVE") if metadata_db else 0,
    }

    return {
        "status": "success" if not failed else "partial",
        "uploaded": success_count,
        "failed": len(failed),
        "failed_details": failed,
        "message": f"Uploaded {success_count} images successfully. They will be used in the next fine-tuning run.",
        "stats": stats
    }


@app.get("/api/get-allowed-classes")
async def get_allowed_classes():
    """Return the list of fixed allowed support classes."""
    return {
        "allowed_classes": ALLOWED_SUPPORT_CLASSES,
        "count": len(ALLOWED_SUPPORT_CLASSES),
    }


@app.get("/api/get-pending-uploads")
async def get_pending_uploads():
    """Return pending uploads grouped by class."""
    if not metadata_db:
        return {"status": "error", "message": "Database not initialized"}
    
    pending_by_class = {cls: [] for cls in ALLOWED_SUPPORT_CLASSES}
    all_pending = metadata_db.get_pending_images()
    
    for img in all_pending:
        if img["class"] in pending_by_class:
            pending_by_class[img["class"]].append({
                "id": img["id"],
                "filename": img["filename"],
                "path": img["path"],
                "class_name": img["class"],
                "created_at": img["created_at"],
            })
    
    total_pending = sum(len(imgs) for imgs in pending_by_class.values())
    
    return {
        "total_pending": total_pending,
        "pending_by_class": pending_by_class,
    }


def _index_image_list(
    paths: list,
    source: str = "TEST",
    filenames: Optional[dict] = None,
    upload_dates: Optional[dict] = None,
) -> None:
    """
    Incrementally index a specific list of image files into Qdrant + SQLite.

    Only processes the given paths — does NOT touch existing indexed images.
    This is O(new_files) instead of O(all_files), making uploads fast.

    Args:
        paths: Absolute file paths to index.
        source: Qdrant/SQLite source label (e.g. "TEST", "SUPPORT").
        filenames: Optional mapping {abs_path: original_filename}.
        upload_dates: Optional mapping {abs_path: ISO-8601 date string}.
    """
    global clip, index, vector_store, metadata_db
    from PIL import Image as PILImage
    from src.backend.region_aware.detector import extract_region_metadata_pil

    filenames = filenames or {}
    upload_dates = upload_dates or {}

    db_entries = []
    for path in paths:
        try:
            img = PILImage.open(path).convert("RGB")
            w, h = img.size

            # Global embedding (multi-scale)
            global_emb = clip.encode_image_multiscale(img, scales=[224, 384])

            # Region embeddings
            regions = extract_region_metadata_pil(
                clip, img,
                scales=[0.3, 0.5, 0.7],
                min_px=32,
                min_area_ratio=0.02,
                max_area_ratio=0.80,
                batch_size=16,
            )

            cls = os.path.basename(os.path.dirname(path)) or "unknown"

            # Write to Qdrant
            if vector_store:
                vector_store.upsert_image(
                    path=path,
                    global_embedding=global_emb,
                    regions=regions,
                    cls=cls,
                    source=source,
                )

            # Update in-memory index
            index[path] = {"global_embedding": global_emb, "regions": regions}

            db_entries.append({
                "path": path, "class": cls, "source": source,
                "width": w, "height": h, "n_regions": len(regions),
                "status": "ACTIVE",
                "image_name": filenames.get(path, os.path.basename(path)),
                "upload_date": upload_dates.get(path),
            })
            print(f"[upload] Indexed: {os.path.basename(path)}")
        except Exception as e:
            print(f"[upload] SKIP {os.path.basename(path)}: {e}")

    if metadata_db and db_entries:
        metadata_db.upsert_images_batch(db_entries)


@app.post("/api/upload-main")
async def upload_main(images: list[UploadFile] = File(...)):
    """Save uploads to test_dataset/ and incrementally index only the new files."""
    from datetime import datetime, timezone

    global clip, index, vector_store, metadata_db
    test_dir = os.path.join(PROJECT_ROOT, "test_dataset")
    os.makedirs(test_dir, exist_ok=True)

    saved_paths = []
    filenames_map = {}
    upload_dates_map = {}
    for img in images:
        path, unique_filename = get_unique_filepath(test_dir, img.filename)
        img.filename = unique_filename
        with open(path, "wb") as f:
            f.write(await img.read())
        saved_paths.append(path)
        filenames_map[path] = img.filename
        upload_dates_map[path] = datetime.now(timezone.utc).isoformat()

    # Only index the newly saved files — fast O(new) instead of O(all)
    _index_image_list(
        saved_paths, source="TEST",
        filenames=filenames_map, upload_dates=upload_dates_map,
    )
    return {"status": "success", "count": len(saved_paths)}


@app.post("/api/upload-test")
async def upload_test(images: list[UploadFile] = File(...)):
    """Save uploads to test_dataset/ and incrementally index only the new files."""
    from datetime import datetime, timezone

    global clip, index, vector_store, metadata_db
    test_dir = os.path.join(PROJECT_ROOT, "test_dataset")
    os.makedirs(test_dir, exist_ok=True)

    saved_paths = []
    filenames_map = {}
    upload_dates_map = {}
    for img in images:
        path, unique_filename = get_unique_filepath(test_dir, img.filename)
        img.filename = unique_filename
        with open(path, "wb") as f:
            f.write(await img.read())
        saved_paths.append(path)
        filenames_map[path] = img.filename
        upload_dates_map[path] = datetime.now(timezone.utc).isoformat()

    _index_image_list(
        saved_paths, source="TEST",
        filenames=filenames_map, upload_dates=upload_dates_map,
    )
    return {"status": "success", "count": len(saved_paths)}


from pydantic import BaseModel
class DeletePayload(BaseModel):
    path: str

@app.post("/api/delete-image")
async def delete_image(payload: DeletePayload):
    global vector_store, metadata_db
    path = payload.path

    # ── Guard: SUPPORT images are read-only ───────────────────────────────
    support_dir = os.path.normpath(os.path.join(PROJECT_ROOT, "cleaned_dataset"))
    if os.path.normpath(path).startswith(support_dir):
        return {
            "status": "error",
            "message": "Support images are read-only. Manage them via the Few-Shot Management page."
        }

    if not path or not os.path.exists(path):
        return {"status": "error", "message": "File not found"}

    try:
        os.remove(path)

        # Remove from Qdrant by path payload filter
        if vector_store:
            vector_store.delete_by_path(path)
        # Remove from SQLite
        if metadata_db:
            metadata_db._execute("DELETE FROM dataset_images WHERE path = ?", (path,))
            metadata_db._execute("DELETE FROM pending_support_images WHERE path = ?", (path,))
            metadata_db.conn.commit()

        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/upload-metadata")
async def get_upload_metadata(
    source: str = None,
    cls: str = None,
    limit: int = 200,
):
    """
    Return upload metadata for all images that were uploaded through the UI.

    Each record contains:
      - image_path   : absolute filesystem path
      - upload_date  : ISO-8601 date/time when the image was uploaded
      - image_name   : original filename
      - image_class  : assigned class label
      - source       : SUPPORT / TEST
      - status       : ACTIVE / PENDING / FAILED
    """
    if not metadata_db:
        return {"status": "error", "message": "Database not initialized", "uploads": []}

    uploads = metadata_db.get_upload_metadata(
        source=source or None,
        cls=cls or None,
        limit=limit,
    )
    return {
        "total": len(uploads),
        "uploads": uploads,
    }

@app.get("/api/models")
async def list_models():
    models = glob.glob(os.path.join(PROJECT_ROOT, "finetuned_clip*.pt"))
    latest_info = get_latest_model_path()
    active_model = latest_info[0] if latest_info else None
    
    return {
        "models": [os.path.basename(m) for m in models],
        "active_model": os.path.basename(active_model) if active_model else "Base CLIP (clip-vit-b-32)",
        "count": len(models)
    }

@app.get("/api/finetune/status")
async def finetune_status():
    """Poll training progress."""
    return _training_state

@app.post("/api/finetune/cancel")
async def finetune_cancel():
    """Request cancellation of an ongoing training job."""
    global _training_state
    if _training_state.get("status") == "running":
        _training_state["cancel_requested"] = True
        return {"status": "success", "message": "Cancellation requested."}
    return {"status": "error", "message": "No training is currently running."}


@app.post("/api/finetune")
async def finetune_model():
    """
    Kick off async finetune + index rebuild in a background thread.
    Returns immediately. Poll /api/finetune/status for progress.
    """
    global _training_state

    if _training_state.get("status") == "running":
        return {"status": "already_running", "message": "Training already in progress"}

    _training_state.update({"status": "running", "message": "Starting...", "progress": "", "result": None, "cancel_requested": False})

    def _run():
        global prototypes, clip, vector_store, metadata_db, index, _training_state
        _do_finetune()

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "message": "Training started in background. Poll /api/finetune/status for progress."}


def _do_finetune():
    """
    Finetune model and atomically activate pending images.
    Runs in a background thread — do NOT call directly from a route.
    """
    global prototypes, clip, vector_store, metadata_db, index, _training_state
    
    if not metadata_db or not clip or not vector_store:
        _training_state.update({"status": "error", "message": "System not fully initialized"})
        return
    
    print("=" * 60)
    print("  TRAINING + INDEX REBUILD (ATOMIC)")
    print("=" * 60)
    
    try:
        # ────────────────────────────────────────────────────────────────
        # STEP 1: Validate pending uploads
        # ────────────────────────────────────────────────────────────────
        support_dir = os.path.join(PROJECT_ROOT, "cleaned_dataset")
        pending_images = metadata_db.get_pending_images()
        print(f"\n[finetune] Found {len(pending_images)} pending uploads to activate")
        
        paths_to_activate = []
        for pending_img in pending_images:
            src_path = pending_img["path"]
            if os.path.exists(src_path):
                paths_to_activate.append(src_path)
                # Mark pending record as processed
                metadata_db.mark_pending_processed(src_path, success=True)
            else:
                print(f"  [!] File not found: {src_path}")
                metadata_db.mark_pending_processed(src_path, success=False, 
                                                 error_msg="File not found")
        
        # ────────────────────────────────────────────────────────────────
        # STEP 2: FINETUNE THE MODEL (Incremental)
        # ────────────────────────────────────────────────────────────────
        _training_state["progress"] = "Fine-tuning model (this may take a few minutes)..."
        print("\n[finetune] Starting fine-tuning...")
        
        # Determine next model version
        latest_model_info = get_latest_model_path()
        if latest_model_info and latest_model_info[0]:
            _, current_ver = latest_model_info
            next_ver = current_ver + 1
        else:
            next_ver = 1
            
        next_model_path = os.path.join(PROJECT_ROOT, f"finetuned_clip{next_ver}.pt" if next_ver > 0 else "finetuned_clip.pt")
        
        # Run training
        try:
            # We already have the latest model loaded in clip (if it existed)
            # finetune_clip updates it in-place and saves the new checkpoint
            finetune_clip(
                clip_encoder=clip,
                support_dir=support_dir,
                epochs=30,  # can be adjusted
                batch_size=8,
                lr_backbone=1e-5,
                lr_head=1e-3,
                save_path=next_model_path,
                cancel_check=lambda: _training_state.get("cancel_requested", False)
            )
            if _training_state.get("cancel_requested", False):
                _training_state.update({"status": "error", "message": "Training cancelled by user."})
                # Revert pending records so they can be trained later
                for pending_img in pending_images:
                    metadata_db._execute("UPDATE pending_support_images SET processed_at = NULL WHERE path = ?", (pending_img["path"],))
                metadata_db.conn.commit()
                print("[finetune] Cancellation acknowledged. Reverted pending status.")
                return
            
            print(f"[finetune] Successfully saved new model: {next_model_path}")
        except Exception as e:
            print(f"[finetune] Training error: {e}")
            raise e

        # ────────────────────────────────────────────────────────────────
        # STEP 3: Build prototypes from updated support set
        # ────────────────────────────────────────────────────────────────
        _training_state["progress"] = "Building class prototypes..."
        print(f"\n[finetune] Rebuilding prototypes...")
        prototypes = build_prototypes(clip, support_dir)
        
        # Save prototypes to SQLite
        metadata_db.clear_prototypes()
        for cls_name, proto_emb in prototypes.items():
            metadata_db.save_prototype(cls_name, proto_emb, n_images=0)
        
        # ────────────────────────────────────────────────────────────────
        # STEP 4: Index all ACTIVE support images to Qdrant
        # ────────────────────────────────────────────────────────────────
        _training_state["progress"] = "Indexing support images into Qdrant..."
        print(f"\n[finetune] Indexing support dataset...")
        
        # Delete old support entries
        vector_store.delete_by_source("SUPPORT")
        metadata_db.delete_images_by_source("SUPPORT")
        
        # Re-index all support classes
        support_index = build_index(
            clip, support_dir,
            cache_path="",
            use_regions=True,
            source="SUPPORT",
            vector_store=vector_store,
            metadata_db=metadata_db,
        )
        
        # ────────────────────────────────────────────────────────────────
        # STEP 4: ATOMIC ACTIVATION - Mark all processed images as ACTIVE
        # ────────────────────────────────────────────────────────────────
        print(f"\n[finetune] Activating {len(paths_to_activate)} newly indexed images...")
        metadata_db.activate_images(paths_to_activate)
        
        # Reload in-memory index
        index = load_index_from_qdrant(vector_store)
        
        print(f"\n[finetune] SUCCESS!")
        print(f"  Qdrant: {vector_store.count()} total points")
        print(f"  Support: {vector_store.count('SUPPORT')} points (ACTIVE)")
        print(f"  Prototypes: {list(prototypes.keys())}")
        
        # ────────────────────────────────────────────────────────────────
        # STEP 5: Post-Training Testing
        # ────────────────────────────────────────────────────────────────
        _training_state["progress"] = "Running automatic post-training validation test..."
        try:
            print("\n[finetune] Running post-training validation test...")
            test_cls = list(prototypes.keys())[0] if prototypes else "heavy drop"
            from src.backend.retrieval.retriever import build_query_embedding, retrieve_with_qdrant
            q_mat, _ = build_query_embedding(clip, text_query=test_cls, use_regions=False)
            test_res = retrieve_with_qdrant(q_mat, vector_store, prototypes, top_k=3, use_regions=False)
            print(f"  [Test] Query: '{test_cls}'")
            for i, r in enumerate(test_res):
                print(f"    {i+1}. {r[2]} (score: {r[1]:.3f}) - {os.path.basename(r[0])}")
        except Exception as e:
            print(f"  [Test] Skipped: {e}")
        
        # Clear processed pending records
        metadata_db.clear_pending_images()
        
        result = {
            "status": "success",
            "message": f"Trained on {len(paths_to_activate)} new images and rebuilt index",
            "newly_activated": len(paths_to_activate),
            "qdrant_support": vector_store.count("SUPPORT"),
            "qdrant_total": vector_store.count(),
            "prototypes": list(prototypes.keys()),
        }
        _training_state.update({"status": "success", "message": result["message"], "result": result})
        return
        
    except Exception as e:
        print(f"\n[finetune] FAILED: {e}")
        print(traceback.format_exc())
        
        # Mark pending images as FAILED
        pending_images = metadata_db.get_pending_images()
        for pending_img in pending_images:
            metadata_db.mark_pending_processed(
                pending_img["path"],
                success=False,
                error_msg=str(e)
            )
        
        result = {
            "status": "error",
            "message": f"Training failed: {str(e)}",
            "pending_images_marked_failed": len(pending_images),
        }
        _training_state.update({"status": "error", "message": result["message"], "result": result})
        return


@app.post("/api/rebuild-index")
async def rebuild_index_api():
    """
    Full rebuild: Re-index all ACTIVE images from all sources (dataset, support, test).
    
    NOTE: Training/activation of pending images is done in /api/finetune.
    This endpoint only rebuilds the index from already ACTIVE images.
    """
    global index, clip, vector_store, metadata_db
    print("=" * 60)
    print("  FULL INDEX REBUILD (ACTIVE IMAGES ONLY)")
    print("=" * 60)

    try:
        # Wipe and recreate index
        vector_store.reset()
        metadata_db.delete_images_by_source("TRAINING")
        metadata_db.delete_images_by_source("SUPPORT")
        metadata_db.delete_images_by_source("TEST")

        # Re-index all three sources
        _index_main_dataset(clip, vector_store, metadata_db)
        _index_support_dataset(clip, vector_store, metadata_db)

        test_dir = os.path.join(PROJECT_ROOT, "test_dataset")
        if os.path.exists(test_dir) and os.listdir(test_dir):
            _index_test_dataset(clip, vector_store, metadata_db)

        # Reload in-memory index
        index = load_index_from_qdrant(vector_store)

        print(f"\n[rebuild] Done. Qdrant: {vector_store.count()} points, "
              f"In-memory: {len(index)} images")

        return {
            "status": "success",
            "points": vector_store.count(),
            "images": len(index),
        }
    except Exception as e:
        print(f"[rebuild] Failed: {e}")
        print(traceback.format_exc())
        return {
            "status": "error",
            "message": str(e),
        }


@app.post("/api/search")
async def search(
    image: UploadFile = File(None),
    text: str = Form(None),
    top_k: int = Form(5),
    mode: str = Form("global"),
    aggregation: str = Form("max"),
    threshold: float = Form(0.5),
    use_regions: str = Form("false"),
    use_finetuned: str = Form("false"),
    use_qdrant: str = Form("true"),
    forced_class: str = Form(None),
    from_date: str = Form(None),
    to_date: str = Form(None),
):
    global clip, index, prototypes, vector_store, metadata_db
    t_start = time.time()
    use_regions_bool = use_regions.lower() == "true"
    use_qdrant_bool = use_qdrant.lower() == "true"

    # Validate: at least one query input (image, text, OR forced_class from dropdown)
    if not image and not text and not forced_class:
        return {"error": "Provide at least an image, text query, or select a class filter", "results": []}

    # Normalize forced_class: empty string -> None
    forced_class = forced_class.strip() if forced_class else None
    if forced_class == "":
        forced_class = None

    # Treat text as forced_class if it case-insensitively matches one of the allowed classes
    if text and not forced_class:
        text_norm = text.strip().lower()
        for allowed in ALLOWED_SUPPORT_CLASSES:
            if allowed.lower() == text_norm:
                forced_class = allowed   # use the canonical class name (correct case)
                text = None
                break

    # If forced_class is set, null out text — prototype vector is used as query instead
    effective_text = None if forced_class else text

    # Save uploaded query image temporarily
    query_path = None
    if image and image.filename:
        query_path = os.path.join(PROJECT_ROOT, f"temp_{image.filename}")
        with open(query_path, "wb") as f:
            f.write(await image.read())

    try:
        # Enforce region-aware rules: text search -> r=NO, both -> r=NO.
        if effective_text and not image:
            use_regions_bool = False
        if effective_text and image:
            use_regions_bool = False

        # Build query embedding (handles image-only, text-only, forced_class, or hybrid)
        query_matrix, _ = build_query_embedding(
            clip,
            image_path=query_path,
            text_query=effective_text,
            use_regions=use_regions_bool,
            forced_class=forced_class,
            prototypes=prototypes,
        )

        from_ts = float(from_date) if from_date else None
        to_ts = float(to_date) if to_date else None

        # Retrieve — use Qdrant ANN or fallback to brute-force
        if use_qdrant_bool and vector_store is not None:
            results = retrieve_with_qdrant(
                query_matrix, vector_store, prototypes,
                top_k=top_k, use_regions=use_regions_bool,
                from_date=from_ts, to_date=to_ts,
                forced_class=forced_class,
            )
        else:
            results = retrieve_with_prototypes(
                query_matrix, index, prototypes,
                top_k=top_k, use_regions=use_regions_bool,
                forced_class=forced_class,
            )
    except Exception as e:
        print(f"[search] ERROR: {e}")
        traceback.print_exc()
        return {"error": str(e), "results": [], "analytics": {"time_ms": 0}}
    finally:
        if query_path and os.path.exists(query_path):
            os.remove(query_path)

    time_ms = int((time.time() - t_start) * 1000)

    # Format results for UI
    formatted_results = []
    for rank, item in enumerate(results):
        path = item[0]
        score = float(item[1])
        cls = item[2] if len(item) > 2 else None
        bbox = item[3] if len(item) > 3 else None

        # Make path accessible via URL
        rel_path = path.replace("\\", "/")
        if not rel_path.startswith("http"):
            if "cleaned_dataset" in rel_path:
                idx = rel_path.find("cleaned_dataset")
                url_path = "/" + rel_path[idx:]
            elif "test_dataset" in rel_path:
                idx = rel_path.find("test_dataset")
                url_path = "/" + rel_path[idx:]
            elif "dataset" in rel_path:
                idx = rel_path.find("dataset")
                url_path = "/" + rel_path[idx:]
            else:
                url_path = rel_path
        else:
            url_path = rel_path

        formatted_results.append({
            "id": os.path.basename(path),
            "image_path": url_path,
            "similarity": score,
            "matched_class": cls,
            "bbox": bbox
        })

    # Log query to SQLite
    if metadata_db:
        metadata_db.log_query(
            query_image=query_path,
            text_query=text,
            top_k=top_k,
            mode=mode,
            use_regions=use_regions_bool,
            results=[{"path": r["image_path"], "score": r["similarity"]} for r in formatted_results],
            time_ms=time_ms,
        )

    return {
        "results": formatted_results,
        "analytics": {
            "time_ms": time_ms,
            "mode": mode,
            "engine": "qdrant" if use_qdrant_bool else "brute_force",
            "top_score": formatted_results[0]["similarity"] if formatted_results else 0.0,
            "total_retrieved": len(formatted_results)
        }
    }


# ── DB stats endpoints ──────────────────────────────────────────────────────

@app.get("/api/stats")
async def get_stats():
    """Return database statistics."""
    db_stats = metadata_db.stats() if metadata_db else {}
    qdrant_stats = {
        "total_points": vector_store.count() if vector_store else 0,
        "dataset_points": vector_store.count("TRAINING") if vector_store else 0,
        "support_points": vector_store.count("SUPPORT") if vector_store else 0,
        "test_points": vector_store.count("TEST") if vector_store else 0,
    }
    return {"sqlite": db_stats, "qdrant": qdrant_stats}


@app.get("/api/query-history")
async def get_query_history(limit: int = 50):
    if not metadata_db:
        return {"history": []}
    return {"history": metadata_db.get_query_history(limit)}


@app.get("/api/finetune-runs")
async def get_finetune_runs(limit: int = 20):
    if not metadata_db:
        return {"runs": []}
    return {"runs": metadata_db.get_finetune_runs(limit)}


# ── DB Explorer endpoints ────────────────────────────────────────────────────

@app.get("/api/db/collections")
async def db_collections():
    """Qdrant collection overview."""
    sources = ["dataset", "support", "test"]
    collections = []
    for src in sources:
        count = vector_store.count(src) if vector_store else 0
        collections.append({"name": src, "points": count})
    return {
        "collections": collections,
        "total": vector_store.count() if vector_store else 0,
        "collection_name": "clip_embeddings",
        "vector_size": 512,
    }


@app.get("/api/db/images")
async def db_images(
    source: str = None,
    cls: str = None,
    limit: int = 100,
    offset: int = 0,
):
    """Browse SQLite image records with optional filters."""
    if not metadata_db:
        return {"images": [], "total": 0}
    all_images = metadata_db.get_images(source=source or None, cls=cls or None)
    total = len(all_images)
    page = all_images[offset: offset + limit]
    for img in page:
        p = img["path"].replace("\\", "/")
        for prefix in ["cleaned_dataset", "test_dataset", "dataset"]:
            if prefix in p:
                img["url"] = "/" + p[p.find(prefix):]
                break
        else:
            img["url"] = None
        img["filename"] = os.path.basename(p)
    return {"images": page, "total": total, "limit": limit, "offset": offset}


@app.get("/api/db/classes")
async def db_classes():
    """List all unique classes and their counts from SQLite."""
    if not metadata_db:
        return {"classes": [], "allowed": ALLOWED_SUPPORT_CLASSES}
    rows = metadata_db.conn.execute(
        "SELECT class, source, COUNT(*) as count FROM dataset_images GROUP BY class, source ORDER BY source, class"
    ).fetchall()
    return {
        "classes": [dict(r) for r in rows],
        "allowed": ALLOWED_SUPPORT_CLASSES,
    }


@app.get("/api/db/prototypes")
async def db_prototypes():
    """List class prototype metadata."""
    if not metadata_db:
        return {"prototypes": []}
    rows = metadata_db.conn.execute(
        "SELECT class_name, n_images, updated_at FROM class_prototypes ORDER BY class_name"
    ).fetchall()
    return {"prototypes": [dict(r) for r in rows]}


@app.delete("/api/db/cleanup-class/{class_name}")
async def db_cleanup_class(class_name: str):
    """
    Remove a non-allowed class from the system.
    Moves images to _archive, removes from SQLite and Qdrant.
    """
    if class_name in ALLOWED_SUPPORT_CLASSES:
        return {"status": "error", "message": f"Cannot delete an allowed class: {class_name}"}

    support_dir = os.path.join(PROJECT_ROOT, "cleaned_dataset")
    class_dir = os.path.join(support_dir, class_name)
    archive_dir = os.path.join(support_dir, "_archive", class_name)

    moved = 0
    if os.path.exists(class_dir):
        os.makedirs(archive_dir, exist_ok=True)
        import shutil
        for f in os.listdir(class_dir):
            shutil.move(os.path.join(class_dir, f), os.path.join(archive_dir, f))
            moved += 1
        os.rmdir(class_dir)

    # Remove from SQLite
    if metadata_db:
        metadata_db.conn.execute("DELETE FROM dataset_images WHERE class = ?", (class_name,))
        metadata_db.conn.commit()

    return {
        "status": "success",
        "message": f"Archived {moved} images from '{class_name}'",
        "archived": moved,
    }


@app.get("/api/stats")
async def get_stats():
    # Provide simple high-level stats for the UI DatasetOverview
    stats = {
        "sqlite": {
            "total_images": metadata_db.image_count(),
            "support_images": metadata_db.image_count("SUPPORT"),
            "test_images": metadata_db.image_count("TEST"),
            "pending_uploads": metadata_db.pending_image_count(),
            "prototypes": len(prototypes)
        }
    }
    return stats

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)