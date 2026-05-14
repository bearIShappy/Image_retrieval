"""
One-shot object detection using CLIP with improved region proposals.

Changes vs original:
  - sliding_window_proposals(): added small scales [0.15, 0.25] for distant /
    small objects; reduced default stride_ratio 0.5->0.3 for finer coverage;
    reduced default min_size 80->48 px.
  - grabcut_mask_crop(): new helper — 2-iteration GrabCut foreground isolation
    on each crop before CLIP encoding, makes embeddings background-invariant.
  - build_support_embeddings(): loads ALL images per class (was: break after
    first), averages their multi-scale embeddings -> richer class prototype.
  - extract_region_metadata() / extract_region_metadata_pil(): added
    use_grabcut flag; tightened default area filters; use updated defaults.
  - detect_regions(): added use_grabcut flag; passes finer scale/stride to
    sliding_window_proposals; removed redundant single-batch branch.
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
from src.backend.retrieval.clip_model import CLIPEncoder


# ── Region metadata type ─────────────────────────────────────────────────────
# Each region dict looks like:
#   {"bbox": (x, y, w, h), "area_ratio": float, "embedding": tensor(1,D)}


# ── GrabCut foreground masking ───────────────────────────────────────────────

def grabcut_mask_crop(pil_crop: Image.Image) -> Image.Image:
    """
    Apply 2-iteration GrabCut to isolate the foreground of a crop.

    Converts background pixels to the mean colour of the crop so the
    CLIP embedding focuses on the subject regardless of what is behind it.

    Falls back to the original crop if GrabCut fails or the crop is too
    small for the algorithm (< 10 px on either side).

    Args:
        pil_crop: PIL RGB crop.

    Returns:
        PIL RGB image with background suppressed.
    """
    w, h = pil_crop.size
    if w < 10 or h < 10:
        return pil_crop

    bgr = cv2.cvtColor(np.array(pil_crop), cv2.COLOR_RGB2BGR)

    # Tight rect leaving a 5 % border
    margin_x = max(2, int(w * 0.05))
    margin_y = max(2, int(h * 0.05))
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    # GrabCut needs the rect dimensions > 0
    if rect[2] <= 0 or rect[3] <= 0:
        return pil_crop

    try:
        mask = np.zeros(bgr.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(bgr, mask, rect, bgd_model, fgd_model,
                    iterCount=2, mode=cv2.GC_INIT_WITH_RECT)

        # Pixels marked as definite / probable foreground
        fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                           1, 0).astype(np.uint8)

        # If foreground is tiny (< 10 % of pixels), skip masking
        if fg_mask.mean() < 0.10:
            return pil_crop

        rgb = np.array(pil_crop)
        # Fill background with the mean foreground colour
        mean_color = rgb[fg_mask == 1].mean(axis=0).astype(np.uint8) \
            if fg_mask.any() else np.array([128, 128, 128], dtype=np.uint8)
        masked = rgb.copy()
        masked[fg_mask == 0] = mean_color
        return Image.fromarray(masked)

    except cv2.error:
        return pil_crop


# ── Region proposals ─────────────────────────────────────────────────────────

def sliding_window_proposals(
    image_shape: Tuple[int, int],
    scales: List[float] = [0.15, 0.25, 0.35, 0.5, 0.7],   # added small scales
    stride_ratio: float = 0.3,                               # finer stride
    min_size: int = 48,                                      # smaller min box
) -> List[Tuple[int, int, int, int]]:
    """
    Multi-scale sliding window proposals.

    Defaults updated vs original:
      scales      : [0.15, 0.25, 0.35, 0.5, 0.7]  (was [0.3, 0.5, 0.7])
      stride_ratio: 0.3                              (was 0.5)
      min_size    : 48 px                            (was 80 px)

    Smaller scales catch distant / small parachutes; finer stride reduces
    missed objects near window boundaries.
    """
    h_img, w_img = image_shape[:2]
    proposals = []
    seen = set()
    for scale in scales:
        win_w = max(int(w_img * scale), min_size)
        win_h = max(int(h_img * scale), min_size)
        sx = max(int(win_w * stride_ratio), 1)
        sy = max(int(win_h * stride_ratio), 1)
        for y in range(0, h_img - win_h + 1, sy):
            for x in range(0, w_img - win_w + 1, sx):
                ar = win_w / win_h
                if 0.3 < ar < 3.0:
                    key = (x, y, win_w, win_h)
                    if key not in seen:
                        seen.add(key)
                        proposals.append(key)
    return proposals


# ── NMS ──────────────────────────────────────────────────────────────────────

def compute_iou(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int],
) -> float:
    """
    Compute Intersection-over-Union for two (x, y, w, h) boxes.

    Args:
        box1: (x, y, w, h) of first box.
        box2: (x, y, w, h) of second box.

    Returns:
        IoU value in [0, 1].
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi = max(x1, x2)
    yi = max(y1, y2)
    xf = min(x1 + w1, x2 + w2)
    yf = min(y1 + h1, y2 + h2)

    inter = max(0, xf - xi) * max(0, yf - yi)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


# Keep old name as alias for backward compat
_iou = compute_iou


def nms(
    detections: list,
    iou_threshold: float = 0.5,
    use_nms: bool = True,
) -> list:
    """
    Class-aware Non-Maximum Suppression.

    Accepts detections as either:
      - Tuples: (x, y, w, h, class_name, score)
      - Dicts:  {"bbox": (x,y,w,h), "score": float, "class": str, ...}

    Groups detections by class, then within each class keeps the
    highest-scoring box and suppresses all boxes with IoU > iou_threshold.

    Args:
        detections: List of detection tuples or dicts.
        iou_threshold: Overlap threshold for suppression (default 0.5).
        use_nms: If False, return detections unchanged (bypass).

    Returns:
        Filtered detections in the same format as input.
    """
    if not detections or not use_nms:
        return detections

    is_dict = isinstance(detections[0], dict)

    def _get_key(det):
        if is_dict:
            return det["bbox"], det["score"], det["class"]
        return det[:4], det[5], det[4]

    by_class: Dict[str, list] = {}
    for det in detections:
        _, _, cls = _get_key(det)
        by_class.setdefault(cls, []).append(det)

    kept = []
    for cls, cls_dets in by_class.items():
        cls_dets = sorted(cls_dets, key=lambda d: _get_key(d)[1], reverse=True)
        selected = []
        while cls_dets:
            best = cls_dets.pop(0)
            selected.append(best)
            best_box = _get_key(best)[0]
            cls_dets = [
                d for d in cls_dets
                if compute_iou(best_box, _get_key(d)[0]) < iou_threshold
            ]
        kept.extend(selected)

    kept.sort(key=lambda d: _get_key(d)[1], reverse=True)
    return kept


# ── Support embeddings ───────────────────────────────────────────────────────

def build_support_embeddings(
    clip: CLIPEncoder,
    support_dir: str,
) -> Dict[str, torch.Tensor]:
    """
    Build per-class support embeddings by averaging ALL images in each class
    folder (original: only the first image was used).

    Multi-scale embedding is computed for each image; all are averaged and
    re-normalised to form a single robust prototype per class.

    Args:
        clip: CLIPEncoder instance.
        support_dir: Root directory with one sub-folder per class.

    Returns:
        Dict of {class_name: averaged_embedding (1, D)}.
    """
    from src.backend.indexing.indexer import IMAGE_EXTENSIONS
    embeddings = {}
    print(f"[detector] Building support embeddings from {support_dir}")

    for cls_name in sorted(os.listdir(support_dir)):
        cls_dir = os.path.join(support_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue

        class_embs = []
        for fname in sorted(os.listdir(cls_dir)):
            if os.path.splitext(fname)[1].lower() not in IMAGE_EXTENSIONS:
                continue
            img = Image.open(os.path.join(cls_dir, fname)).convert("RGB")
            emb = clip.encode_image_multiscale(img)   # (1, D)
            class_embs.append(emb)
            print(f"  [+] {cls_name:<35s} <- {fname}")

        if not class_embs:
            continue

        # Average across all support images -> single normalised prototype
        stacked = torch.cat(class_embs, dim=0)         # (N_imgs, D)
        mean_emb = stacked.mean(dim=0, keepdim=True)   # (1, D)
        mean_emb = mean_emb / mean_emb.norm(dim=-1, keepdim=True)
        embeddings[cls_name] = mean_emb
        print(f"  [OK] {cls_name}: {len(class_embs)} image(s) averaged")

    print(f"[detector] {len(embeddings)} support classes ready.")
    return embeddings


# ── Region metadata extraction ───────────────────────────────────────────────

def extract_region_metadata(
    clip: CLIPEncoder,
    image_path: str,
    scales: List[float] = [0.15, 0.25, 0.35, 0.5, 0.7],
    stride_ratio: float = 0.3,
    min_px: int = 48,
    min_area_ratio: float = 0.01,   # lowered from 0.02 to catch small objects
    max_area_ratio: float = 0.80,
    batch_size: int = 16,
    use_grabcut: bool = True,
) -> List[dict]:
    """
    Extract region proposals with metadata for indexing/retrieval.

    For each valid region returns:
        {"bbox": (x, y, w, h), "area_ratio": float, "embedding": tensor(1,D)}

    Changes vs original:
      - Finer default scales / stride / min_px.
      - use_grabcut: applies grabcut_mask_crop() before CLIP encoding so
        embeddings are background-invariant.

    Args:
        clip: CLIPEncoder instance.
        image_path: Path to image file.
        scales: Sliding window scale fractions.
        stride_ratio: Stride as fraction of window size.
        min_px: Minimum region dimension in pixels.
        min_area_ratio: Minimum region area / image area.
        max_area_ratio: Maximum region area / image area.
        batch_size: CLIP inference batch size.
        use_grabcut: Apply GrabCut foreground masking before encoding.

    Returns:
        List of region metadata dicts.
    """
    img = Image.open(image_path).convert("RGB")
    w_img, h_img = img.size
    img_area = w_img * h_img

    proposals = sliding_window_proposals(
        (h_img, w_img), scales=scales, stride_ratio=stride_ratio, min_size=min_px
    )

    boxes = []
    for (x, y, bw, bh) in proposals:
        if bw < min_px or bh < min_px:
            continue
        ratio = (bw * bh) / img_area
        if ratio < min_area_ratio or ratio > max_area_ratio:
            continue
        boxes.append((x, y, bw, bh, ratio))

    if not boxes:
        return []

    # Crop (+ optional GrabCut masking)
    crops = []
    for (x, y, bw, bh, _) in boxes:
        crop = img.crop((x, y, min(x + bw, w_img), min(y + bh, h_img)))
        if use_grabcut:
            crop = grabcut_mask_crop(crop)
        crops.append(crop)

    all_embs = []
    for i in range(0, len(crops), batch_size):
        embs = clip.encode_image_batch(crops[i:i + batch_size])
        all_embs.append(embs)
    crop_embs = torch.cat(all_embs, dim=0)  # (N, D)

    regions = []
    for i, (x, y, bw, bh, ratio) in enumerate(boxes):
        regions.append({
            "bbox": (x, y, bw, bh),
            "area_ratio": round(ratio, 4),
            "embedding": crop_embs[i:i + 1],  # (1, D)
        })

    return regions


def extract_region_metadata_pil(
    clip: CLIPEncoder,
    image: Image.Image,
    scales: List[float] = [0.15, 0.25, 0.35, 0.5, 0.7],
    stride_ratio: float = 0.3,
    min_px: int = 48,
    min_area_ratio: float = 0.01,
    max_area_ratio: float = 0.80,
    batch_size: int = 16,
    use_grabcut: bool = True,
) -> List[dict]:
    """
    Same as extract_region_metadata but accepts a PIL image directly.
    Used for query images that are already loaded in memory.
    """
    w_img, h_img = image.size
    img_area = w_img * h_img

    proposals = sliding_window_proposals(
        (h_img, w_img), scales=scales, stride_ratio=stride_ratio, min_size=min_px
    )

    boxes = []
    for (x, y, bw, bh) in proposals:
        if bw < min_px or bh < min_px:
            continue
        ratio = (bw * bh) / img_area
        if ratio < min_area_ratio or ratio > max_area_ratio:
            continue
        boxes.append((x, y, bw, bh, ratio))

    if not boxes:
        return []

    crops = []
    for (x, y, bw, bh, _) in boxes:
        crop = image.crop((x, y, min(x + bw, w_img), min(y + bh, h_img)))
        if use_grabcut:
            crop = grabcut_mask_crop(crop)
        crops.append(crop)

    all_embs = []
    for i in range(0, len(crops), batch_size):
        embs = clip.encode_image_batch(crops[i:i + batch_size])
        all_embs.append(embs)
    crop_embs = torch.cat(all_embs, dim=0)

    regions = []
    for i, (x, y, bw, bh, ratio) in enumerate(boxes):
        regions.append({
            "bbox": (x, y, bw, bh),
            "area_ratio": round(ratio, 4),
            "embedding": crop_embs[i:i + 1],
        })

    return regions


# ── Detection ────────────────────────────────────────────────────────────────

def detect_regions(
    clip: CLIPEncoder,
    image_path: str,
    support_embeddings: Dict[str, torch.Tensor],
    threshold: float = 0.25,
    nms_iou: float = 0.5,
    min_region: int = 48,           # was 80; lowered to catch small/distant targets
    batch_size: int = 16,
    max_detections: int = 15,
    use_nms: bool = True,
    use_grabcut: bool = True,       # new: foreground masking per crop
) -> Tuple[np.ndarray, List[Tuple[int, int, int, int, str, float]]]:
    """
    One-shot detection: region proposals -> (GrabCut) -> CLIP encode -> match -> NMS.

    Changes vs original:
      - Passes finer scales/stride to sliding_window_proposals.
      - use_grabcut: applies grabcut_mask_crop() before encoding each crop,
        making detections background-invariant.
      - min_region lowered from 80 to 48 px.

    Args:
        threshold: Minimum CLIP similarity to keep a detection.
        nms_iou: IoU threshold for NMS (default 0.5).
        min_region: Minimum box dimension in pixels.
        batch_size: CLIP inference batch size.
        max_detections: Cap output to top-N detections.
        use_nms: If True apply class-aware NMS, if False skip NMS.
        use_grabcut: Apply GrabCut foreground masking before encoding.

    Returns:
        (image_rgb, detections) where detections = [(x,y,w,h,class,score),...].
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Cannot read: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img = img_rgb.shape[:2]
    pil_img = Image.fromarray(img_rgb)

    # Use finer proposals (updated defaults in sliding_window_proposals)
    proposals = sliding_window_proposals((h_img, w_img))
    print(f"[detector] {len(proposals)} proposals "
          f"(scales=[0.15,0.25,0.35,0.5,0.7], stride=0.3)")

    # Filter, crop, optionally mask
    boxes, crops = [], []
    for (x, y, w, h) in proposals:
        if w < min_region or h < min_region:
            continue
        crop = pil_img.crop((x, y, min(x + w, w_img), min(y + h, h_img)))
        if crop.size[0] < 10 or crop.size[1] < 10:
            continue
        if use_grabcut:
            crop = grabcut_mask_crop(crop)
        boxes.append((x, y, w, h))
        crops.append(crop)

    print(f"[detector] {len(crops)} valid crops "
          f"(grabcut={'on' if use_grabcut else 'off'})")
    if not crops:
        return img_rgb, []

    # Batch encode
    crop_embs = torch.cat(
        [clip.encode_image_batch(crops[i:i + batch_size])
         for i in range(0, len(crops), batch_size)],
        dim=0,
    )

    # Compare with support prototypes
    cls_names = list(support_embeddings.keys())
    ref_stack = torch.cat([support_embeddings[c] for c in cls_names], dim=0)
    sim_matrix = crop_embs @ ref_stack.T        # (N_crops, N_classes)
    best_scores, best_idxs = sim_matrix.max(dim=1)

    results = []
    for i in range(len(boxes)):
        score = best_scores[i].item()
        if score >= threshold:
            cls = cls_names[best_idxs[i].item()]
            results.append((*boxes[i], cls, round(score, 4)))

    count_before = len(results)
    print(f"[detector] {count_before} detections above threshold={threshold}")

    results = nms(results, iou_threshold=nms_iou, use_nms=use_nms)
    count_after_nms = len(results)

    results = sorted(results, key=lambda d: d[5], reverse=True)[:max_detections]

    if use_nms:
        print(f"[detector] NMS: {count_before} -> {count_after_nms} "
              f"(iou_thresh={nms_iou}) -> {len(results)} after cap")
    else:
        print(f"[detector] NMS disabled, {len(results)} after cap")

    return img_rgb, results