"""
Few-shot fine-tuning for CLIP ViT visual backbone.

Fine-tunes the last 2 transformer blocks + final LayerNorm of the
CLIP vision encoder using a small labelled support set.  Weights are
updated IN-PLACE on the CLIPEncoder so that all existing encode_*
methods automatically use the fine-tuned model.

Dependencies: torch, torchvision, PIL (no extra packages).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from PIL import Image
from typing import Dict, Optional, Tuple

from src.backend.retrieval.clip_model import CLIPEncoder

# ── Constants ────────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

# CLIP ViT-B/32 normalization (from preprocessor_config.json)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


# ═════════════════════════════════════════════════════════════════════════════
# 1. FewShotDataset
# ═════════════════════════════════════════════════════════════════════════════

class FewShotDataset(Dataset):
    """
    Few-shot classification dataset that scans class sub-folders
    automatically and applies strong augmentation.

    Directory layout expected::

        support_dir/
        ├── class_a/
        │   ├── img1.jpg
        │   └── img2.png
        └── class_b/
            └── img3.jpg

    Returns
    -------
    (image_tensor, label_int)
        image_tensor : torch.Tensor  (3, 224, 224)
        label_int    : int
    """

    def __init__(self, support_dir: str) -> None:
        super().__init__()
        self.samples: list = []          # [(image_path, label_int), ...]
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}

        # Scan class folders
        class_names = sorted(
            d for d in os.listdir(support_dir)
            if os.path.isdir(os.path.join(support_dir, d))
        )
        for idx, cls_name in enumerate(class_names):
            self.class_to_idx[cls_name] = idx
            self.idx_to_class[idx] = cls_name
            cls_dir = os.path.join(support_dir, cls_name)
            for fname in sorted(os.listdir(cls_dir)):
                if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                    self.samples.append(
                        (os.path.join(cls_dir, fname), idx)
                    )

        if not self.samples:
            raise ValueError(
                f"No images found in {support_dir}. "
                "Expected sub-folders with images."
            )

        print(f"[finetune] FewShotDataset: {len(self.samples)} images, "
              f"{len(self.class_to_idx)} classes")
        for cls_name in self.class_to_idx:
            idx = self.class_to_idx[cls_name]
            n = sum(1 for _, l in self.samples if l == idx)
            print(f"  [{idx}] {cls_name}: {n} images")

        # Strong augmentation pipeline (PIL → Tensor)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3,
                saturation=0.2, hue=0.1,
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        return tensor, label


# ═════════════════════════════════════════════════════════════════════════════
# 2. finetune_clip
# ═════════════════════════════════════════════════════════════════════════════

def finetune_clip(
    clip_encoder: CLIPEncoder,
    support_dir: str,
    epochs: int = 30,
    batch_size: int = 8,
    lr_backbone: float = 1e-5,
    lr_head: float = 1e-3,
    weight_decay: float = 1e-4,
    save_path: str = "finetuned_clip.pt",
    device: Optional[str] = None,
) -> CLIPEncoder:
    """
    Fine-tune the CLIP visual backbone on a few-shot support set.

    Strategy:
      - Freeze ALL layers except the last 2 transformer blocks and
        the final LayerNorm of the vision encoder.
      - Add a linear classification head.
      - Train with CrossEntropyLoss (label smoothing 0.1).
      - AdamW with separate lr for backbone vs. head.
      - CosineAnnealingLR scheduler.
      - Mixed precision (float16) when CUDA is available.

    After training the visual encoder state dict is written back
    IN-PLACE to ``clip_encoder.model`` so every ``encode_*`` method
    automatically uses the fine-tuned weights.

    Parameters
    ----------
    clip_encoder : CLIPEncoder
        Existing encoder instance whose visual backbone will be updated.
    support_dir : str
        Path to the support dataset root (class sub-folders).
    epochs : int
        Training epochs (default 30).
    batch_size : int
        Batch size (default 8).
    lr_backbone : float
        Learning rate for unfrozen backbone layers (default 1e-5).
    lr_head : float
        Learning rate for classification head (default 1e-3).
    weight_decay : float
        Weight decay for AdamW (default 1e-4).
    save_path : str
        Where to save the checkpoint.
    device : str or None
        Device override; auto-detected if None.

    Returns
    -------
    CLIPEncoder
        The same encoder instance with updated weights.
    """
    device = device or clip_encoder.device

    # ── Dataset & loader ─────────────────────────────────────────────────
    dataset = FewShotDataset(support_dir)
    num_classes = len(dataset.class_to_idx)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,          # safe for Windows
        drop_last=False,
        pin_memory=(device != "cpu"),
    )

    # ── Extract visual encoder components ────────────────────────────────
    # HuggingFace CLIPModel structure:
    #   model.vision_model.embeddings
    #   model.vision_model.pre_layrnorm
    #   model.vision_model.encoder.layers[0..11]   (12 blocks for ViT-B)
    #   model.vision_model.post_layernorm
    #   model.visual_projection                     (768 → 512)
    clip_model = clip_encoder.model
    vision_model = clip_model.vision_model
    encoder_layers = vision_model.encoder.layers
    num_layers = len(encoder_layers)

    # ── Freeze everything first ──────────────────────────────────────────
    for param in clip_model.parameters():
        param.requires_grad = False

    # ── Unfreeze last 2 transformer blocks ───────────────────────────────
    layers_to_unfreeze = encoder_layers[num_layers - 2:]   # last 2 blocks
    for layer in layers_to_unfreeze:
        for param in layer.parameters():
            param.requires_grad = True

    # ── Unfreeze final LayerNorm ─────────────────────────────────────────
    for param in vision_model.post_layernorm.parameters():
        param.requires_grad = True

    # ── Classification head ──────────────────────────────────────────────
    # The visual projection maps 768 → 512; we classify from 512-D.
    embed_dim = clip_model.config.projection_dim   # 512
    classifier = nn.Linear(embed_dim, num_classes).to(device)

    # ── Param groups ─────────────────────────────────────────────────────
    backbone_params = []
    for layer in layers_to_unfreeze:
        backbone_params.extend(layer.parameters())
    backbone_params.extend(vision_model.post_layernorm.parameters())

    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone,
             "weight_decay": weight_decay},
            {"params": classifier.parameters(), "lr": lr_head,
             "weight_decay": weight_decay},
        ],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Mixed precision ──────────────────────────────────────────────────
    use_amp = (device != "cpu") and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ── Training loop ────────────────────────────────────────────────────
    clip_model.train()
    classifier.train()

    print(f"\n[finetune] Starting fine-tuning on {device}")
    print(f"  epochs={epochs}  batch_size={batch_size}  "
          f"lr_backbone={lr_backbone}  lr_head={lr_head}")
    print(f"  unfrozen: last {len(list(layers_to_unfreeze))} transformer blocks "
          f"+ post_layernorm")
    print(f"  classes: {num_classes}  embed_dim: {embed_dim}")
    print(f"  mixed precision: {'ON' if use_amp else 'OFF'}\n")

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    features = clip_model.get_image_features(
                        pixel_values=images
                    )
                    # Handle cases where output is an object instead of a raw tensor
                    if not isinstance(features, torch.Tensor):
                        features = features.pooler_output if hasattr(features, "pooler_output") else features[0]
                    
                    logits = classifier(features)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                features = clip_model.get_image_features(
                    pixel_values=images
                )
                # Handle cases where output is an object instead of a raw tensor
                if not isinstance(features, torch.Tensor):
                    features = features.pooler_output if hasattr(features, "pooler_output") else features[0]
                
                logits = classifier(features)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  Epoch {epoch:>3}/{epochs}  loss={avg_loss:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

    # ── Restore eval mode ────────────────────────────────────────────────
    clip_model.eval()
    classifier.eval()

    # ── Freeze everything again (inference) ──────────────────────────────
    for param in clip_model.parameters():
        param.requires_grad = False

    # ── The weights are already updated IN-PLACE in clip_encoder.model ───
    # No need to copy anything; the encoder's encode_* methods will
    # automatically use the fine-tuned weights.

    # ── Save checkpoint ──────────────────────────────────────────────────
    checkpoint = {
        "visual_state_dict": vision_model.state_dict(),
        "visual_projection_state_dict": clip_model.visual_projection.state_dict(),
        "classifier_state_dict": classifier.state_dict(),
        "class_to_idx": dataset.class_to_idx,
    }
    torch.save(checkpoint, save_path)
    print(f"\n[finetune] Checkpoint saved -> {save_path}")
    print(f"[finetune] Fine-tuning complete.  Encoder weights updated in-place.")

    return clip_encoder


def load_finetuned_weights(
    clip_encoder: CLIPEncoder,
    checkpoint_path: str = "finetuned_clip.pt",
    device: Optional[str] = None,
) -> Dict[str, int]:
    """
    Load previously saved fine-tuned weights into the CLIPEncoder IN-PLACE.

    Parameters
    ----------
    clip_encoder : CLIPEncoder
        Encoder whose visual backbone will be updated.
    checkpoint_path : str
        Path to the saved checkpoint (.pt file).
    device : str or None
        Device override.

    Returns
    -------
    class_to_idx : Dict[str, int]
        Mapping from class name to integer label.
    """
    device = device or clip_encoder.device
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load visual backbone weights
    clip_encoder.model.vision_model.load_state_dict(
        checkpoint["visual_state_dict"]
    )

    # Load visual projection weights if available
    if "visual_projection_state_dict" in checkpoint:
        clip_encoder.model.visual_projection.load_state_dict(
            checkpoint["visual_projection_state_dict"]
        )

    clip_encoder.model.eval()
    class_to_idx = checkpoint.get("class_to_idx", {})

    print(f"[finetune] Loaded fine-tuned weights from {checkpoint_path}")
    print(f"  classes: {list(class_to_idx.keys())}")

    return class_to_idx


# ═════════════════════════════════════════════════════════════════════════════
# 3. build_prototypes
# ═════════════════════════════════════════════════════════════════════════════

def build_prototypes(
    clip: CLIPEncoder,
    support_dir: str,
) -> Dict[str, torch.Tensor]:
    """
    Build class prototypes by averaging (and L2-normalising) image
    embeddings from the support set.

    For each class folder every image is encoded via
    ``clip.encode_image_pil`` and the resulting embeddings are averaged
    then L2-normalized.

    Parameters
    ----------
    clip : CLIPEncoder
        Encoder (ideally already fine-tuned).
    support_dir : str
        Root of the support dataset with class sub-folders.

    Returns
    -------
    Dict[str, torch.Tensor]
        ``{class_name: tensor(1, D)}`` — one L2-normalized prototype
        per class.  Compatible with ``retrieve_with_prototypes()``.
    """
    prototypes: Dict[str, torch.Tensor] = {}

    class_names = sorted(
        d for d in os.listdir(support_dir)
        if os.path.isdir(os.path.join(support_dir, d))
    )

    print(f"[finetune] Building prototypes from {support_dir}")

    for cls_name in class_names:
        cls_dir = os.path.join(support_dir, cls_name)
        embeddings = []

        for fname in sorted(os.listdir(cls_dir)):
            if os.path.splitext(fname)[1].lower() not in IMAGE_EXTENSIONS:
                continue
            img_path = os.path.join(cls_dir, fname)
            try:
                img = Image.open(img_path).convert("RGB")
                emb = clip.encode_image_pil(img)      # (1, D)
                embeddings.append(emb)
            except Exception as e:
                print(f"  [WARN] Skipping {fname}: {e}")

        if not embeddings:
            print(f"  [SKIP] {cls_name}: no valid images found")
            continue

        # Average embeddings → (1, D)
        stacked = torch.cat(embeddings, dim=0)         # (N, D)
        mean_emb = stacked.mean(dim=0, keepdim=True)   # (1, D)

        # L2 normalize
        mean_emb = mean_emb / mean_emb.norm(dim=-1, keepdim=True)

        prototypes[cls_name] = mean_emb
        print(f"  [OK] {cls_name}: {len(embeddings)} images -> prototype {mean_emb.shape}")

    print(f"[finetune] {len(prototypes)} class prototypes ready.\n")
    return prototypes
