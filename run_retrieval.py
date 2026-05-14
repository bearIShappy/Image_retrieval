"""
End-to-end CLIP few-shot retrieval pipeline.

Orchestrates:
  1. CLIP model loading (auto-applies finetuned_clip.pt if present)
  2. Optional fine-tuning, or loading saved weights explicitly
  3. Prototype building
  4. Index building / loading (Qdrant + SQLite or legacy pickle)
  5. Query embedding construction (image, text, or hybrid)
  6. Prototype-boosted retrieval (ANN via Qdrant or brute-force)
  7. Visualization

Usage examples::

    # Image query (fine-tuned weights auto-loaded from finetuned_clip.pt):
    python run_retrieval.py --query path/to/query.jpg --dataset ./cleaned_dataset

    # Text-only query (no query image needed):
    python run_retrieval.py --text "cargo parachute" --text_only --dataset ./cleaned_dataset

    # Hybrid image + text:
    python run_retrieval.py --query img.jpg --text "cargo parachute" --dataset ./cleaned_dataset

    # Force fine-tuning even if checkpoint exists:
    python run_retrieval.py --query img.jpg --dataset ./cleaned_dataset --force_finetune

    # Region-aware scoring:
    python run_retrieval.py --query img.jpg --dataset ./cleaned_dataset --use_regions
"""

import os
import sys
import time
import argparse

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.backend.retrieval.clip_model import CLIPEncoder
from src.backend.indexing.indexer import build_index, load_index, load_index_from_qdrant
from src.backend.retrieval.retriever import build_query_embedding, retrieve_with_prototypes, retrieve_with_qdrant
from src.backend.region_aware.visualize import show_retrieval_results
from src.backend.training.finetune import (
    finetune_clip,
    load_finetuned_weights,
    build_prototypes,
)
from src.backend.metadata.vector_store import VectorStore, migrate_pkl_to_qdrant
from src.backend.metadata.metadata_db import MetadataDB


# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "clip-vit-b-32")
DEFAULT_DATASET    = os.path.join(PROJECT_ROOT, "cleaned_dataset")
DEFAULT_SUPPORT    = os.path.join(PROJECT_ROOT, "cleaned_dataset")
DEFAULT_CACHE      = os.path.join(PROJECT_ROOT, "embeddings.pkl")
DEFAULT_CHECKPOINT = os.path.join(PROJECT_ROOT, "finetuned_clip.pt")
DEFAULT_OUTPUT     = os.path.join(PROJECT_ROOT, "output.png")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the retrieval pipeline.

    Returns:
        argparse.Namespace: Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="CLIP Few-Shot Retrieval Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Query inputs (at least one of --query or --text is required) ────────
    parser.add_argument(
        "--query", type=str, default=None,
        help="Path to query image (omit for text-only search).",
    )
    parser.add_argument(
        "--text", type=str, default=None,
        help="Optional text query to combine with (or replace) image query.",
    )
    parser.add_argument(
        "--text_only", action="store_true",
        help="Perform text-only retrieval.  --query is ignored and --text is required.",
    )

    # ── Dataset & paths ───────────────────────────────────────────────────
    parser.add_argument(
        "--dataset", type=str, default=DEFAULT_DATASET,
        help="Root directory of the dataset to search against.",
    )
    parser.add_argument(
        "--support", type=str, default=DEFAULT_SUPPORT,
        help="Root directory of the support set (class sub-folders).",
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Number of retrieval results to return (default: 5).",
    )
    parser.add_argument(
        "--use_regions", action="store_true",
        help="Enable region-aware index and scoring.",
    )

    # ── Model & checkpoint ────────────────────────────────────────────────
    parser.add_argument(
        "--model_path", type=str, default=DEFAULT_MODEL_PATH,
        help="Path to local CLIP model directory.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
        help="Path to fine-tuned checkpoint file (default: finetuned_clip.pt).",
    )
    parser.add_argument(
        "--cache", type=str, default=DEFAULT_CACHE,
        help="Path to embeddings cache (embeddings.pkl).",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT,
        help="Path to save visualization output.",
    )

    # ── Fine-tuning control ──────────────────────────────────────────────
    parser.add_argument(
        "--skip_finetune", action="store_true",
        help="Explicitly skip fine-tuning and load checkpoint weights.",
    )
    parser.add_argument(
        "--force_finetune", action="store_true",
        help="Force fine-tuning even if a checkpoint already exists.",
    )
    parser.add_argument(
        "--no_qdrant", action="store_true",
        help="Disable Qdrant; use legacy pickle + brute-force.",
    )

    # ── Fine-tuning hyper-parameters ─────────────────────────────────────
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr_backbone",type=float, default=1e-5)
    parser.add_argument("--lr_head",    type=float, default=1e-3)

    return parser.parse_args()


# ── Pretty printer ───────────────────────────────────────────────────────────

def print_results_table(results, top_k: int) -> None:
    """Print ranked retrieval results to console."""
    print(f"\n{'═' * 95}")
    print(f"  Top-{top_k} Retrieval Results")
    print(f"{'═' * 95}")
    print(f"  {'Rank':<6} {'Score':<10} {'Class':<30} {'Path'}")
    print(f"  {'─' * 4:<6} {'─' * 7:<10} {'─' * 25:<30} {'─' * 40}")

    for rank, item in enumerate(results, 1):
        path  = item[0]
        score = item[1]
        cls   = item[2] if len(item) > 2 else "—"
        bbox  = item[3] if len(item) > 3 else None

        basename = os.path.basename(path)
        line = f"  {rank:<6} {score:<10.4f} {cls:<30} {basename}"
        if bbox is not None:
            line += f"  bbox={bbox}"
        print(line)

    print(f"{'═' * 95}\n")


# ── Main pipeline ───────────────────────────────────────────────────────────

def main() -> None:
    """Run the end-to-end CLIP retrieval pipeline."""
    args = parse_args()

    # ── Input validation ─────────────────────────────────────────────────────
    if args.text_only:
        if not args.text:
            print("[ERROR] --text_only requires --text <query string>.")
            sys.exit(1)
        args.query = None  # ensure no image path leaks in
        print(f"[pipeline] Mode: TEXT-ONLY  query='{args.text}'")
    elif args.query:
        if not os.path.isfile(args.query):
            print(f"[ERROR] Query image not found: {args.query}")
            sys.exit(1)
        mode = "HYBRID" if args.text else "IMAGE"
        print(f"[pipeline] Mode: {mode}  query={args.query}"
              + (f"  text='{args.text}'" if args.text else ""))
    else:
        print("[ERROR] Provide at least --query <image> or --text <string> (with --text_only).")
        sys.exit(1)

    t_start = time.time()
    use_qdrant = not args.no_qdrant

    # ── Auto-detect skip_finetune ──────────────────────────────────────────────
    # If neither --skip_finetune nor --force_finetune is set, auto-detect:
    # skip if the checkpoint already exists (weights already in CLIPEncoder).
    if not args.force_finetune and not args.skip_finetune:
        if os.path.isfile(args.checkpoint):
            print(
                f"[pipeline] Checkpoint found at '{args.checkpoint}'. "
                "Auto-skipping fine-tuning (use --force_finetune to override)."
            )
            args.skip_finetune = True
        else:
            print(
                "[pipeline] No checkpoint found. Fine-tuning will run. "
                "Use --skip_finetune to load an explicit checkpoint or "
                "--force_finetune to suppress this warning."
            )

    # ── 1. Load pre-trained CLIPEncoder ─────────────────────────────────────
    print("\n[pipeline] Step 1: Loading CLIP model")
    clip = CLIPEncoder(model_path=args.model_path)
    # NOTE: CLIPEncoder.__init__ already applies finetuned_clip.pt if found.
    # Step 2 below only re-loads weights explicitly when the user passes
    # --skip_finetune (different checkpoint path) or triggers fresh training.

    # ── 1b. Initialize stores ────────────────────────────────────────────────
    vector_store = None
    metadata_db = None
    if use_qdrant:
        print("\n[pipeline] Step 1b: Initializing Qdrant + SQLite")
        vector_store = VectorStore()
        metadata_db = MetadataDB()

    # ── 2. Fine-tune or load checkpoint ────────────────────────────────────
    # CLIPEncoder already applied the default checkpoint during __init__.
    # Here we handle only edge cases: different checkpoint path, or forced training.
    if args.skip_finetune:
        if args.checkpoint != DEFAULT_CHECKPOINT:
            # User pointed to a non-default checkpoint — load it explicitly.
            print(f"\n[pipeline] Step 2: Loading weights from '{args.checkpoint}'")
            if not os.path.isfile(args.checkpoint):
                print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
                sys.exit(1)
            load_finetuned_weights(clip, args.checkpoint)
        else:
            print(
                "\n[pipeline] Step 2: Fine-tuned weights already applied by "
                "CLIPEncoder (default checkpoint path)."
            )
    else:
        # Fresh fine-tuning requested (--force_finetune or no checkpoint found).
        print("\n[pipeline] Step 2: Fine-tuning CLIP visual backbone")
        clip = finetune_clip(
            clip_encoder=clip,
            support_dir=args.support,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr_backbone=args.lr_backbone,
            lr_head=args.lr_head,
            save_path=args.checkpoint,
        )

    # ── 3. Build prototypes ──────────────────────────────────────────────
    print("\n[pipeline] Step 3: Building class prototypes")
    prototypes = build_prototypes(clip, args.support)

    # Save prototypes to SQLite
    if metadata_db:
        for cls_name, proto_emb in prototypes.items():
            metadata_db.save_prototype(cls_name, proto_emb, n_images=0)

    # ── 4. Load or build index ───────────────────────────────────────────
    print("\n[pipeline] Step 4: Loading / building dataset index")
    index = None

    if use_qdrant and vector_store.count() > 0:
        print("  Loading index from Qdrant...")
        index = load_index_from_qdrant(vector_store)
    elif use_qdrant and os.path.isfile(args.cache):
        print("  Migrating pickle to Qdrant...")
        migrate_pkl_to_qdrant(args.cache, vector_store, source="dataset")
        index = load_index_from_qdrant(vector_store)
    else:
        if os.path.isfile(args.cache):
            index = load_index(args.cache)

    if index is None:
        print(f"  Cache not found → building index from {args.dataset}")
        index = build_index(
            clip, args.dataset,
            cache_path=args.cache,
            use_multiscale=True,
            use_regions=args.use_regions,
            source="dataset",
            vector_store=vector_store,
            metadata_db=metadata_db,
        )

    if not index:
        print("[ERROR] No images indexed. Check --dataset directory.")
        sys.exit(1)

    print(f"  Index contains {len(index)} images")
    if use_qdrant:
        print(f"  Qdrant points: {vector_store.count()}")

    # ── 5. Build query embedding ─────────────────────────────────────────
    _query_desc = args.query if args.query else f"(text-only: '{args.text}')"
    print(f"\n[pipeline] Step 5: Building query embedding — {_query_desc}")
    query_matrix, debug = build_query_embedding(
        clip,
        image_path=args.query,       # None when --text_only
        text_query=args.text,
        use_regions=args.use_regions and bool(args.query),  # regions need an image
    )
    print(f"  Query matrix shape: {query_matrix.shape}")
    for k, v in debug.items():
        if k != "query_regions":
            print(f"  {k}: {v}")

    # ── 6. Retrieval ─────────────────────────────────────────────────────
    print(f"\n[pipeline] Step 6: Retrieving top-{args.top_k} results")

    if use_qdrant and vector_store is not None:
        print("  Engine: Qdrant ANN")
        results = retrieve_with_qdrant(
            query_matrix, vector_store, prototypes,
            top_k=args.top_k, use_regions=args.use_regions,
        )
    else:
        print("  Engine: brute-force (legacy)")
        results = retrieve_with_prototypes(
            query_matrix, index, prototypes,
            top_k=args.top_k, use_regions=args.use_regions,
        )

    print_results_table(results, args.top_k)

    # ── 7. Visualization ─────────────────────────────────────────────────
    if args.query:
        print(f"[pipeline] Step 7: Saving visualization → {args.output}")
        show_retrieval_results(
            query_path=args.query,
            results=results,
            save_path=args.output,
        )
    else:
        print(
            "[pipeline] Step 7: Skipping visualization (no query image in text-only mode). "
            f"Results written to console above."
        )

    elapsed = time.time() - t_start
    print(f"\n[pipeline] Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
