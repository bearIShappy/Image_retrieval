"""
Advanced retriever with multi-region query matching.

Supports both legacy index ({path: tensor}) and extended index
({path: {global_embedding, regions}}).

NEW: Qdrant-backed search via VectorStore for ~10x faster ANN.

Region-aware scoring:
  final = 0.7 * max_region_similarity + 0.3 * global_similarity

Returns matched bbox when region scoring is used.
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
from src.backend.retrieval.clip_model import CLIPEncoder
from src.backend.indexing.indexer import get_global_embedding, get_regions, is_extended_index


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between a (1,D) and b (N,D) -> (N,)"""
    return (a @ b.T).squeeze(0)


def _region_boxes(
    w: int, h: int,
    scales: List[float] = [0.4, 0.6, 0.8, 1.0],
    stride_ratio: float = 0.5,
    min_size: int = 100,
) -> List[Tuple[int, int, int, int]]:
    """Generate multi-scale region boxes for query analysis."""
    boxes = []
    for scale in scales:
        bw = max(int(w * scale), min_size)
        bh = max(int(h * scale), min_size)
        if bw > w or bh > h:
            continue
        sx = max(int(bw * stride_ratio), 1)
        sy = max(int(bh * stride_ratio), 1)
        for y in range(0, h - bh + 1, sy):
            for x in range(0, w - bw + 1, sx):
                boxes.append((x, y, bw, bh))
    return boxes


# ── Query embedding ──────────────────────────────────────────────────────────

def build_query_embedding(
    clip: CLIPEncoder,
    image_path: Optional[str] = None,
    text_query: Optional[str] = None,
    image_weight: float = 0.7,
    use_regions: bool = True,
    use_multiscale: bool = True,
    max_regions: int = 30,
    forced_class: Optional[str] = None,
    prototypes: Optional[Dict] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Build a rich query representation from image, text, or both.

    Supports:
      - Image-only: full + multiscale + region embeddings
      - Text-only: text embedding
      - Hybrid: image embeddings + weighted image-text blend

    Returns:
        (query_matrix, debug_info)
        query_matrix: (K, D) tensor of all query embeddings
        debug_info: dict with intermediate results
    """
    if not image_path and not text_query and not forced_class:
        raise ValueError("At least one of image_path, text_query, or forced_class must be provided")

    debug = {}
    embeddings = []
    full_emb = None

    # ── Image branch ─────────────────────────────────────────────────────
    if image_path:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        debug["image_size"] = (w, h)

        # 1. Full image embedding
        full_emb = clip.encode_image_pil(img)
        embeddings.append(full_emb)
        debug["full_emb_shape"] = full_emb.shape

        # 2. Multi-scale
        if use_multiscale:
            ms_emb = clip.encode_image_multiscale(img, scales=[224, 384, 512])
            embeddings.append(ms_emb)

        # 3. Region embeddings with metadata
        query_regions = []
        if use_regions:
            from src.backend.region_aware.detector import extract_region_metadata_pil
            region_metas = extract_region_metadata_pil(
                clip, img,
                scales=[0.3, 0.5, 0.7],
                min_px=32,
                min_area_ratio=0.02,
                max_area_ratio=0.80,
                batch_size=16,
            )
            if len(region_metas) > max_regions:
                idxs = np.linspace(0, len(region_metas) - 1, max_regions, dtype=int)
                region_metas = [region_metas[i] for i in idxs]

            for rm in region_metas:
                embeddings.append(rm["embedding"])
                query_regions.append(rm)
            debug["n_regions"] = len(region_metas)

        debug["query_regions"] = query_regions
    else:
        debug["query_regions"] = []

    # ── Text / forced-class branch ───────────────────────────────────────
    if forced_class and prototypes:
        # Normalize lookup: "static_line_jump" -> matches "Static Line Jump" etc.
        def _norm_key(k):
            return k.lower().replace(" ", "_").replace("-", "_")

        proto_key = None
        forced_norm = _norm_key(forced_class)
        for k in prototypes:
            if _norm_key(k) == forced_norm:
                proto_key = k
                break

        if proto_key is not None:
            proto_emb = prototypes[proto_key]
            if full_emb is not None:
                p = proto_emb[:1] if proto_emb.dim() == 2 else proto_emb
                hybrid = image_weight * full_emb + (1 - image_weight) * p
                hybrid = hybrid / hybrid.norm(dim=-1, keepdim=True)
                embeddings.append(hybrid)
            else:
                embeddings.append(proto_emb)
            debug["text_query"] = f"[forced_class={proto_key}]"
            debug["proto_key_resolved"] = proto_key
        else:
            print(f"[retriever] WARNING: forced_class='{forced_class}' not in prototypes {list(prototypes.keys())}")
    elif text_query:
        text_emb = clip.encode_text(text_query)
        if full_emb is not None:
            # Hybrid: blend image + text
            hybrid = image_weight * full_emb + (1 - image_weight) * text_emb
            hybrid = hybrid / hybrid.norm(dim=-1, keepdim=True)
            embeddings.append(hybrid)
        else:
            # Text-only: use text embedding directly
            embeddings.append(text_emb)
        debug["text_query"] = text_query

    # Safety guard: if embeddings is still empty, nothing matched — return zero vector
    if not embeddings:
        print(f"[retriever] WARNING: no embeddings built — check forced_class key or prototype dict")
        embeddings.append(torch.zeros(1, 512))

    query_matrix = torch.cat(embeddings, dim=0)  # (K, D)
    debug["total_query_vectors"] = query_matrix.shape[0]

    return query_matrix, debug


# ── Legacy retrieval (unchanged API) ─────────────────────────────────────────

def retrieve(
    query_embedding: torch.Tensor,
    index: Dict[str, Any],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """
    Basic single-vector retrieval using cosine similarity.
    Works with both legacy and extended index formats.
    """
    if not index:
        return []

    paths = list(index.keys())
    embs = torch.cat([get_global_embedding(index[p]) for p in paths], dim=0)
    scores = _cosine_sim(query_embedding, embs)

    k = min(top_k, len(paths))
    top_scores, top_idxs = scores.topk(k)
    return [(paths[idx.item()], top_scores[i].item()) for i, idx in enumerate(top_idxs)]


def retrieve_multiquery(
    query_matrix: torch.Tensor,
    index: Dict[str, Any],
    top_k: int = 10,
    aggregation: str = "max",
) -> List[Tuple[str, float]]:
    """
    Multi-query retrieval: compare ALL query vectors against each dataset image,
    take the MAX similarity per dataset image.
    Works with both legacy and extended index formats (uses global embedding).
    """
    if not index:
        return []

    paths = list(index.keys())
    db_embs = torch.cat([get_global_embedding(index[p]) for p in paths], dim=0)

    sim_matrix = query_matrix @ db_embs.T  # (K, N)

    if aggregation == "max":
        scores = sim_matrix.max(dim=0).values
    else:
        scores = sim_matrix.mean(dim=0)

    k = min(top_k, len(paths))
    top_scores, top_idxs = scores.topk(k)
    return [(paths[idx.item()], top_scores[i].item()) for i, idx in enumerate(top_idxs)]


# ── Region-aware retrieval ───────────────────────────────────────────────────

def _region_score_for_image(
    query_matrix: torch.Tensor,
    entry: Any,
    debug: bool = False,
) -> Tuple[float, float, Optional[Tuple[int, int, int, int]]]:
    """
    Compute region-aware score for a single index entry.

    Returns:
        (global_sim, max_region_sim, best_bbox)
    """
    global_emb = get_global_embedding(entry)  # (1, D)
    global_sim = (query_matrix @ global_emb.T).max().item()

    regions = get_regions(entry)
    if not regions:
        return global_sim, global_sim, None

    region_embs = torch.cat([r["embedding"] for r in regions], dim=0)
    region_sims = query_matrix @ region_embs.T
    best_flat = region_sims.argmax().item()
    best_region_idx = best_flat % region_sims.shape[1]
    max_region_sim = region_sims.max().item()
    best_bbox = regions[best_region_idx]["bbox"]

    return global_sim, max_region_sim, best_bbox


def retrieve_region_aware(
    query_matrix: torch.Tensor,
    index: Dict[str, Any],
    top_k: int = 10,
    region_weight: float = 0.7,
    global_weight: float = 0.3,
    debug: bool = False,
) -> List[Tuple[str, float, Optional[Tuple[int, int, int, int]]]]:
    """
    Region-aware retrieval scoring.

    For each dataset image:
        final = region_weight * max_region_sim + global_weight * global_sim
    """
    if not index:
        return []

    scored = []
    for path, entry in index.items():
        global_sim, region_sim, best_bbox = _region_score_for_image(
            query_matrix, entry, debug=debug
        )
        final = region_weight * region_sim + global_weight * global_sim
        scored.append((path, final, best_bbox))

        if debug:
            name = path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
            print(f"  [debug] {name[:40]:<40s}  "
                  f"global={global_sim:.4f}  region={region_sim:.4f}  "
                  f"final={final:.4f}  bbox={best_bbox}")

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ── Qdrant-backed retrieval (NEW) ────────────────────────────────────────────

def retrieve_with_qdrant(
    query_matrix: torch.Tensor,
    vector_store,
    prototypes: Dict[str, torch.Tensor],
    top_k: int = 10,
    use_regions: bool = True,
    debug: bool = False,
    from_date: float = None,
    to_date: float = None,
    forced_class: Optional[str] = None,
) -> List[Tuple[str, float, str, Optional[Tuple[int, int, int, int]]]]:
    """
    Qdrant-accelerated retrieval with source-aware searching.

    When a class is identified (forced or auto-detected), the search
    prioritises SUPPORT and TEST sources — the curated images that belong
    to known classes.  TRAINING points (bulk-migrated from pickle) are
    only used as a fallback to fill remaining top_k slots.

    Steps:
      1. Determine the best matching class (forced or via prototypes)
      2. Search SUPPORT + TEST images first (class-filtered when forced)
      3. Optionally search TRAINING for additional matches
      4. Aggregate per-image scores and apply prototype boosting

    Returns:
        List of (path, score, matched_class, best_bbox) sorted descending.
    """
    import os

    # ── Step 1: Class selection ───────────────────────────────────────────
    def _norm_key(k: str) -> str:
        return k.lower().replace(" ", "_").replace("-", "_")

    if forced_class:
        forced_norm = _norm_key(forced_class)
        best_class = next((k for k in prototypes if _norm_key(k) == forced_norm), None)
        best_class_score = 1.0
        print(f"[retriever] Forced class: '{forced_class}' -> resolved to '{best_class}'")
    else:
        best_class, best_class_score = None, -1.0
        for cls_name, proto_matrix in prototypes.items():
            sim = (query_matrix @ proto_matrix.T).max().item()
            if sim > best_class_score:
                best_class_score = sim
                best_class = cls_name
        print(f"[retriever] Best class match: {best_class} (score={best_class_score:.4f})")

    # ── Step 2: Source-aware ANN search ──────────────────────────────────
    # When a class is known, search SUPPORT + TEST first (these have
    # reliable class labels from folder structure).  Then fill remaining
    # slots from TRAINING.

    filter_cls = best_class if forced_class else None
    search_limit = top_k * 4

    all_global_hits: List[dict] = []
    all_region_hits: List[dict] = []

    # Priority 1: SUPPORT images (curated per-class)
    support_globals = vector_store.search_multi(
        query_matrix, top_k=search_limit, globals_only=True,
        from_date=from_date, to_date=to_date,
        class_filter=filter_cls, source_filter="SUPPORT",
    )
    all_global_hits.extend(support_globals)
    if debug:
        print(f"[retriever] SUPPORT global hits: {len(support_globals)}")

    if use_regions:
        support_regions = vector_store.search_multi(
            query_matrix, top_k=search_limit, regions_only=True,
            from_date=from_date, to_date=to_date,
            class_filter=filter_cls, source_filter="SUPPORT",
        )
        all_region_hits.extend(support_regions)

    # Priority 2: TEST images
    test_globals = vector_store.search_multi(
        query_matrix, top_k=search_limit, globals_only=True,
        from_date=from_date, to_date=to_date,
        source_filter="TEST",
    )
    all_global_hits.extend(test_globals)
    if debug:
        print(f"[retriever] TEST global hits: {len(test_globals)}")

    if use_regions:
        test_regions = vector_store.search_multi(
            query_matrix, top_k=search_limit, regions_only=True,
            from_date=from_date, to_date=to_date,
            source_filter="TEST",
        )
        all_region_hits.extend(test_regions)

    # Priority 3: TRAINING images (bulk dataset — fill remaining slots)
    if not forced_class:
        training_globals = vector_store.search_multi(
            query_matrix, top_k=search_limit, globals_only=True,
            from_date=from_date, to_date=to_date,
            source_filter="TRAINING",
        )
        all_global_hits.extend(training_globals)
        if debug:
            print(f"[retriever] TRAINING global hits: {len(training_globals)}")

        if use_regions:
            training_regions = vector_store.search_multi(
                query_matrix, top_k=search_limit, regions_only=True,
                from_date=from_date, to_date=to_date,
                source_filter="TRAINING",
            )
            all_region_hits.extend(training_regions)

    print(
        f"[retriever] Total hits: {len(all_global_hits)} globals, "
        f"{len(all_region_hits)} regions"
    )

    # ── Step 3: Aggregate per image ──────────────────────────────────────
    per_image: Dict[str, dict] = {}
    for h in all_global_hits:
        p = h["path"]
        if p not in per_image or h["score"] > per_image[p]["global_score"]:
            per_image[p] = {
                "global_score": h["score"],
                "region_score": per_image.get(p, {}).get("region_score", 0.0),
                "best_bbox": per_image.get(p, {}).get("best_bbox"),
                "class": h.get("class", "unknown"),
                "source": h.get("source", "TRAINING"),
            }

    for h in all_region_hits:
        p = h["path"]
        if p not in per_image:
            per_image[p] = {
                "global_score": 0.0,
                "region_score": h["score"],
                "best_bbox": h.get("bbox"),
                "class": h.get("class", "unknown"),
                "source": h.get("source", "TRAINING"),
            }
        elif h["score"] > per_image[p]["region_score"]:
            per_image[p]["region_score"] = h["score"]
            per_image[p]["best_bbox"] = h.get("bbox")

    # ── Step 4: Score & re-rank ──────────────────────────────────────────
    scored = []
    proto = prototypes.get(best_class) if best_class else None

    for path, info in per_image.items():
        region_s = info["region_score"] if use_regions and info["region_score"] > 0 else info["global_score"]
        base_score = 0.7 * region_s + 0.3 * info["global_score"]

        # Boost SUPPORT/TEST results from the matched class
        source = info.get("source", "TRAINING")
        img_class = info.get("class", "unknown")

        if forced_class and best_class:
            # When class is explicitly forced, boost same-class images
            if _norm_key(img_class) == _norm_key(best_class):
                base_score *= 1.15  # 15% boost for exact class match
            # Boost SUPPORT/TEST over TRAINING
            if source in ("SUPPORT", "TEST"):
                base_score *= 1.10  # 10% boost for curated sources

        resolved_class = best_class if best_class else img_class
        scored.append((path, base_score, resolved_class, info["best_bbox"], source))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Final output: (path, score, class, bbox)
    reranked = [
        (path, score, cls, bbox)
        for path, score, cls, bbox, _src in scored[:top_k]
    ]

    if debug or True:  # always log top results for now
        print(f"[retriever] Top {min(len(reranked), 5)} results:")
        for i, (p, s, c, _) in enumerate(reranked[:5]):
            print(f"  {i+1}. [{c}] score={s:.4f} — {os.path.basename(p)}")

    return reranked


# ── Prototype-boosted retrieval (original, for backward compat) ──────────────

def retrieve_with_prototypes(
    query_matrix: torch.Tensor,
    index: Dict[str, Any],
    prototypes: Dict[str, torch.Tensor],
    top_k: int = 10,
    use_regions: bool = True,
    debug: bool = False,
    forced_class: Optional[str] = None,
) -> List[Tuple[str, float, str, Optional[Tuple[int, int, int, int]]]]:
    """
    Retrieve dataset images using direct similarity + prototype re-ranking.

    If the index has regions AND use_regions is True, uses region-aware scoring.
    Otherwise falls back to global-only scoring.

    Returns:
        List of (path, score, matched_class, best_bbox) sorted descending.
        best_bbox is None when region scoring was not used.
    """
    # Step 1: Class selection (forced from dropdown or computed via CLIP)
    def _norm_key(k):
        return k.lower().replace(" ", "_").replace("-", "_")

    if forced_class:
        forced_norm = _norm_key(forced_class)
        best_class = next((k for k in prototypes if _norm_key(k) == forced_norm), None)
        best_class_score = 1.0
        print(f"[retriever] Forced class: '{forced_class}' -> resolved to '{best_class}'")
    else:
        best_class = None
        best_class_score = -1.0
        for cls_name, proto_matrix in prototypes.items():
            sim = (query_matrix @ proto_matrix.T).max().item()
            if sim > best_class_score:
                best_class_score = sim
                best_class = cls_name
        print(f"[retriever] Best class match: {best_class} (score={best_class_score:.4f})")

    # Step 2: Score all images
    has_regions = use_regions and is_extended_index(index)

    if has_regions:
        print("[retriever] Using region-aware scoring (0.7 region + 0.3 global)")
        raw_results = retrieve_region_aware(
            query_matrix, index, top_k=top_k * 2,
            region_weight=0.7, global_weight=0.3, debug=debug,
        )
    else:
        raw = retrieve_multiquery(query_matrix, index, top_k=top_k * 2, aggregation="max")
        raw_results = [(p, s, None) for p, s in raw]

    # Step 3: Re-rank with prototype affinity
    reranked = []
    import os
    if best_class and best_class in prototypes:
        proto = prototypes[best_class]
        for path, score, bbox in raw_results:
            if forced_class:
                parent_dir = os.path.basename(os.path.dirname(path))
                if _norm_key(parent_dir) != _norm_key(best_class):
                    continue
            db_emb = get_global_embedding(index[path])
            proto_sim = (db_emb @ proto.T).max().item()
            final = 0.6 * score + 0.4 * proto_sim
            reranked.append((path, final, best_class, bbox))
    else:
        reranked = [(p, s, "unknown", bbox) for p, s, bbox in raw_results]

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[:top_k]


# ── Simple hybrid retrieval (unchanged API) ──────────────────────────────────

def retrieve_hybrid(
    image_embedding: torch.Tensor,
    text_embedding: torch.Tensor,
    index: Dict[str, Any],
    top_k: int = 5,
    image_weight: float = 0.7,
) -> List[Tuple[str, float]]:
    """Simple hybrid retrieval with weighted image+text."""
    combined = image_weight * image_embedding + (1 - image_weight) * text_embedding
    combined = combined / combined.norm(dim=-1, keepdim=True)
    return retrieve(combined, index, top_k)