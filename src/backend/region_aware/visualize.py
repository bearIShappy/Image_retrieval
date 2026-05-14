"""
Visualization: retrieval results, detection boxes, debug views.

Extended: show_retrieval_results now draws best matching bbox on
retrieved images when bbox data is available.

Uses Agg backend (non-blocking, saves to file).
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional

COLORS = ["#00e5ff", "#ff6d00", "#76ff03", "#d500f9",
          "#ffea00", "#ff1744", "#00e676", "#2979ff"]


def _color(cls: str, classes: list) -> str:
    idx = classes.index(cls) if cls in classes else 0
    return COLORS[idx % len(COLORS)]


def show_retrieval_results(
    query_path: str,
    results: list,
    save_path: str = "retrieval_result.png",
    title: str = "CLIP Retrieval Results",
) -> None:
    """
    Display query + top-k retrieved images with scores.

    Results can be:
      - (path, score)
      - (path, score, class)
      - (path, score, class, bbox)   <-- draws bbox on matched image

    When bbox is present, the best matching region is highlighted with
    a colored rectangle overlay on the retrieved image.
    """
    n = min(len(results), 5)
    fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 5))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    if not hasattr(axes, "__len__"):
        axes = [axes]

    # Query
    try:
        axes[0].imshow(Image.open(query_path).convert("RGB"))
    except Exception:
        axes[0].text(0.5, 0.5, "Query", ha="center", va="center")
    axes[0].set_title("QUERY", fontsize=12, fontweight="bold", color="#00e5ff")
    axes[0].axis("off")

    # Results
    for i in range(n):
        ax = axes[i + 1]
        item = results[i]
        path = item[0]
        score = item[1]
        cls = item[2] if len(item) > 2 else ""
        bbox = item[3] if len(item) > 3 else None

        try:
            ax.imshow(Image.open(path).convert("RGB"))
        except Exception:
            ax.text(0.5, 0.5, "Error", ha="center", va="center")

        # Draw best matching bbox if available
        if bbox is not None:
            x, y, w, h = bbox
            color = "#ff1744"
            rect = plt.Rectangle(
                (x, y), w, h, fill=False,
                edgecolor=color, linewidth=2.5, linestyle="--",
            )
            ax.add_patch(rect)
            ax.text(
                x + 2, y + 14, f"match",
                color="white", fontsize=7, fontweight="bold",
                bbox=dict(facecolor=color, alpha=0.7, edgecolor="none", pad=1),
            )

        label = f"#{i+1} sim={score:.3f}"
        if cls:
            label += f"\n[{cls}]"
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Retrieval saved -> {save_path}")


def show_detections(
    image: np.ndarray,
    detections: List[Tuple[int, int, int, int, str, float]],
    save_path: str = "detection_result.png",
    title: str = "One-Shot CLIP Detection",
    max_boxes: int = 15,
) -> None:
    """Draw bounding boxes with class labels. Limits to top max_boxes."""
    # Sort by score and limit
    detections = sorted(detections, key=lambda d: d[5], reverse=True)[:max_boxes]

    fig, ax = plt.subplots(1, figsize=(14, 10))
    ax.imshow(image)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    all_cls = sorted(set(d[4] for d in detections))

    for (x, y, w, h, cls, score) in detections:
        c = _color(cls, all_cls)
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor=c, linewidth=2.5)
        ax.add_patch(rect)
        ax.text(x, y - 4, f"{cls} {score:.2f}", color="white", fontsize=9,
                fontweight="bold", bbox=dict(facecolor=c, alpha=0.8, edgecolor="none", pad=2))

    if all_cls:
        patches = [mpatches.Patch(color=_color(c, all_cls), label=c) for c in all_cls]
        ax.legend(handles=patches, loc="upper right", fontsize=10, framealpha=0.8)

    ax.axis("off")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Detection saved -> {save_path}")


def show_debug_regions(
    image_path: str,
    query_debug: dict,
    save_path: str = "debug_regions.png",
) -> None:
    """
    Show debug info about query encoding.

    If query_regions are present in debug, draws them on the image
    with their area_ratio labels.
    """
    img = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(1, figsize=(12, 10))
    ax.imshow(img)

    # Draw query regions if present
    query_regions = query_debug.get("query_regions", [])
    if query_regions:
        for i, rm in enumerate(query_regions):
            x, y, w, h = rm["bbox"]
            ratio = rm["area_ratio"]
            color = COLORS[i % len(COLORS)]
            rect = plt.Rectangle(
                (x, y), w, h, fill=False,
                edgecolor=color, linewidth=1.5, alpha=0.6,
            )
            ax.add_patch(rect)
            ax.text(
                x + 2, y + 12, f"r{i} {ratio:.2f}",
                color="white", fontsize=6,
                bbox=dict(facecolor=color, alpha=0.5, edgecolor="none", pad=1),
            )

    # Build info text (skip non-printable items)
    info_items = {k: v for k, v in query_debug.items()
                  if k not in ("query_regions",)}
    info_text = "\n".join(f"{k}: {v}" for k, v in info_items.items())
    ax.set_title(f"Query Debug ({len(query_regions)} regions)\n{info_text}",
                 fontsize=10, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Debug saved -> {save_path}")


def show_support_gallery(support_dir: str, save_path: str = "support_gallery.png") -> None:
    """Display one image per class from support directory."""
    from src.backend.indexing.indexer import IMAGE_EXTENSIONS
    classes = []
    for name in sorted(os.listdir(support_dir)):
        cdir = os.path.join(support_dir, name)
        if os.path.isdir(cdir):
            for f in sorted(os.listdir(cdir)):
                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                    classes.append((name, os.path.join(cdir, f)))
                    break
    if not classes:
        return
    n = len(classes)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    fig.suptitle("Support Images (1-Shot References)", fontsize=16, fontweight="bold")
    if n == 1:
        axes = [axes]
    for ax, (cls, path) in zip(axes, classes):
        ax.imshow(Image.open(path).convert("RGB"))
        ax.set_title(cls, fontsize=12, fontweight="bold",
                     color=_color(cls, [c[0] for c in classes]))
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Support gallery saved -> {save_path}")
