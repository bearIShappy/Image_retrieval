"""
CLIP model loading and embedding extraction using HuggingFace transformers.

Features:
  - Local model loading (no internet)
  - Auto-loads finetuned_clip.pt weights when present (structured checkpoint)
  - Multi-scale image encoding
  - Region-level encoding
  - Support augmentation (flip, crop, brightness)
  - L2-normalized embeddings

Checkpoint format (saved by finetune.py)::

    {
        "visual_state_dict": <vision_model state dict>,
        "visual_projection_state_dict": <projection layer state dict>,
        "classifier_state_dict": <linear head state dict>,  # not loaded here
        "class_to_idx": {class_name: int, ...},
    }
"""

import os
import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import List, Optional, Tuple
from transformers import CLIPModel, CLIPProcessor, CLIPConfig

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "clip-vit-b-32"
)

# HuggingFace Hub ID used as fallback when the local folder is absent.
# No internet is needed at runtime if the model is already in the HF cache
# (~/.cache/huggingface or %USERPROFILE%\.cache\huggingface on Windows).
_HF_MODEL_ID = "openai/clip-vit-base-patch32"


class CLIPEncoder:
    """HuggingFace CLIP wrapper with advanced encoding strategies."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, device: Optional[str] = None):
        """
        Load the CLIP processor and model.

        Loading priority:
          1. ``model_path`` directory — used when it exists on disk (fully offline).
          2. HuggingFace cache (``openai/clip-vit-base-patch32``) — used when the
             local directory is absent but the model was previously downloaded.
          3. ``finetuned_clip.pt`` used as the *sole* source — architecture is
             reconstructed from a hardcoded ViT-B/32 config and all weights are
             loaded from the checkpoint.  No internet access required.

        Args:
            model_path (str): Absolute path to the local model directory.
            device (str, optional): ``"cuda"`` or ``"cpu"``.  Auto-detected when None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Locate finetuned_clip.pt  — file is at:
        #   src/backend/retrieval/clip_model.py  (3 levels deep inside project root)
        root_dir = os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
        )
        ft_path = os.path.join(root_dir, "finetuned_clip.pt")

        # ── Resolve base model source ─────────────────────────────────────────
        if os.path.isdir(model_path):
            # Path 1: local offline directory
            load_from: str = model_path
            print(f"[clip] Loading base model from local directory: {load_from}")
            self.processor = CLIPProcessor.from_pretrained(load_from)
            self.model = CLIPModel.from_pretrained(load_from).to(self.device)
            print(f"[clip] Base model loaded on {self.device}")

        else:
            # Try the HuggingFace cache before falling back to the checkpoint
            from huggingface_hub import try_to_load_from_cache
            cached = try_to_load_from_cache(_HF_MODEL_ID, "config.json")

            if cached is not None:
                # Path 2: HF cache (offline, previously downloaded)
                print(
                    f"[clip] Local directory not found. "
                    f"Loading from HuggingFace cache: '{_HF_MODEL_ID}'"
                )
                self.processor = CLIPProcessor.from_pretrained(_HF_MODEL_ID)
                self.model = CLIPModel.from_pretrained(_HF_MODEL_ID).to(self.device)
                print(f"[clip] Base model loaded on {self.device}")

            elif os.path.exists(ft_path):
                # Path 3: reconstruct architecture from finetuned_clip.pt alone
                print(
                    f"[clip] No local model dir and no HF cache found.\n"
                    f"[clip] Bootstrapping ViT-B/32 architecture from "
                    f"finetuned_clip.pt (fully offline)."
                )
                self.processor, self.model = self._bootstrap_from_checkpoint(
                    ft_path, self.device
                )
                # Weights already applied — skip the second load below
                self.model.eval()
                print("[clip] Model ready (bootstrapped from checkpoint).")
                return  # early return — no second ft load needed

            else:
                raise RuntimeError(
                    "[clip] Cannot load CLIP model:\n"
                    f"  - Local model dir not found : '{model_path}'\n"
                    f"  - HuggingFace cache empty   : '{_HF_MODEL_ID}'\n"
                    f"  - finetuned_clip.pt missing : '{ft_path}'\n"
                    "Place either the model directory or finetuned_clip.pt in "
                    "the project root."
                )

        # ── Apply fine-tuned weights on top of base (paths 1 & 2) ─────────────
        if os.path.exists(ft_path):
            print(f"[clip] Found fine-tuned checkpoint: {ft_path}")
            self._load_structured_checkpoint(ft_path)
        else:
            print("[clip] No finetuned_clip.pt found — using base CLIP weights.")

        self.model.eval()
        print("[clip] Model ready.")


    # ── Checkpoint helpers ───────────────────────────────────────────────────

    @staticmethod
    def _bootstrap_from_checkpoint(
        checkpoint_path: str,
        device: str,
    ):
        """
        Build a CLIP ViT-B/32 model entirely from ``finetuned_clip.pt``.

        Used when neither a local model directory nor the HuggingFace cache
        is available.  The processor is constructed from a hardcoded vocab /
        config that matches ``openai/clip-vit-base-patch32``; the model
        architecture is initialised from the matching ``CLIPConfig`` and then
        the visual weights are overwritten from the checkpoint.

        Args:
            checkpoint_path (str): Absolute path to ``finetuned_clip.pt``.
            device (str): Target device (``"cuda"`` or ``"cpu"``).

        Returns:
            Tuple[CLIPProcessor, CLIPModel]: Ready-to-use processor and model.

        Raises:
            KeyError: If the checkpoint does not contain ``visual_state_dict``.
        """
        from transformers import (
            CLIPVisionConfig, CLIPTextConfig,
            AutoProcessor,
        )

        # ── Build the ViT-B/32 config that matches openai/clip-vit-base-patch32
        vision_cfg = CLIPVisionConfig(
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            image_size=224,
            patch_size=32,
            projection_dim=512,
        )
        text_cfg = CLIPTextConfig(
            hidden_size=512,
            intermediate_size=2048,
            num_hidden_layers=12,
            num_attention_heads=8,
            max_position_embeddings=77,
            vocab_size=49408,
            projection_dim=512,
        )
        cfg = CLIPConfig(
            vision_config=vision_cfg,
            text_config=text_cfg,
            projection_dim=512,
        )

        model = CLIPModel(cfg).to(device)

        # ── Load visual weights from checkpoint ──────────────────────────────
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        if "visual_state_dict" not in checkpoint:
            raise KeyError(
                "[clip] finetuned_clip.pt is missing 'visual_state_dict'. "
                "Cannot bootstrap model without a local model directory."
            )

        model.vision_model.load_state_dict(
            checkpoint["visual_state_dict"], strict=True
        )
        print("[clip] vision_model loaded from checkpoint.")

        if "visual_projection_state_dict" in checkpoint:
            model.visual_projection.load_state_dict(
                checkpoint["visual_projection_state_dict"], strict=True
            )
            print("[clip] visual_projection loaded from checkpoint.")

        # ── Build processor ──────────────────────────────────────────────────
        # Priority: HF cache  →  open_clip tokenizer (always bundled offline)
        try:
            processor = CLIPProcessor.from_pretrained(_HF_MODEL_ID)
            print("[clip] Processor loaded from HuggingFace cache.")
        except Exception:
            # open_clip ships its BPE vocab file inside the package — no
            # internet or extra downloads required.
            print(
                "[clip] HF processor unavailable. "
                "Building processor from open_clip (offline)."
            )
            import open_clip as _oc
            from transformers import CLIPImageProcessor

            _oc_tokenizer = _oc.get_tokenizer("ViT-B-32")

            class _OpenClipTokenizerWrapper:
                """Thin shim that makes open_clip's tokenizer look like an
                HF tokenizer for the purposes of CLIPProcessor calls."""

                def __init__(self, tok):
                    self._tok = tok
                    self.model_max_length = 77

                def __call__(self, text, **kwargs):
                    import torch
                    if isinstance(text, str):
                        text = [text]
                    tokens = self._tok(text)  # (B, 77) int tensor
                    return {
                        "input_ids": tokens,
                        "attention_mask": (tokens != 0).long(),
                    }

                def __getattr__(self, name):
                    return getattr(self._tok, name)

            image_proc = CLIPImageProcessor(
                size={"shortest_edge": 224},
                crop_size={"height": 224, "width": 224},
                do_center_crop=True,
                do_normalize=True,
                image_mean=[0.48145466, 0.4578275, 0.40821073],
                image_std=[0.26862954, 0.26130258, 0.27577711],
            )

            # Build a minimal CLIPProcessor-compatible object
            class _MinimalProcessor:
                """Minimal processor compatible with CLIPEncoder usage."""

                def __init__(self, img_proc, tok):
                    self.image_processor = img_proc
                    self.tokenizer = tok

                def __call__(self, images=None, text=None,
                             return_tensors="pt", padding=True, **kw):
                    import torch
                    out = {}
                    if images is not None:
                        pv = self.image_processor(
                            images=images,
                            return_tensors=return_tensors,
                        )
                        out.update(pv)
                    if text is not None:
                        tv = self.tokenizer(
                            text if isinstance(text, list) else [text]
                        )
                        out.update({
                            k: v for k, v in tv.items()
                        })
                    return out

                def to(self, device):
                    return self

            processor = _MinimalProcessor(
                image_proc, _OpenClipTokenizerWrapper(_oc_tokenizer)
            )
            print("[clip] Processor built from open_clip (fully offline).")

        return processor, model

    def _load_structured_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load a structured checkpoint produced by ``finetune.py``.

        Applies ``visual_state_dict`` to ``model.vision_model`` and
        ``visual_projection_state_dict`` to ``model.visual_projection``.
        The ``classifier_state_dict`` is intentionally skipped — the
        linear classification head is only used during training and is
        not part of the encoder interface.

        Args:
            checkpoint_path (str): Absolute path to the ``.pt`` checkpoint.

        Raises:
            RuntimeError: Propagated from PyTorch if state-dict shapes mismatch.
        """
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )

            # ── Visual backbone ──────────────────────────────────────────────
            if "visual_state_dict" in checkpoint:
                msg = self.model.vision_model.load_state_dict(
                    checkpoint["visual_state_dict"], strict=True
                )
                print(
                    f"[clip] vision_model loaded  "
                    f"(missing={len(msg.missing_keys)}, "
                    f"unexpected={len(msg.unexpected_keys)})"
                )
            else:
                print("[clip] WARNING: 'visual_state_dict' key missing in checkpoint.")

            # ── Visual projection (768 → 512) ────────────────────────────────
            if "visual_projection_state_dict" in checkpoint:
                msg = self.model.visual_projection.load_state_dict(
                    checkpoint["visual_projection_state_dict"], strict=True
                )
                print(
                    f"[clip] visual_projection loaded  "
                    f"(missing={len(msg.missing_keys)}, "
                    f"unexpected={len(msg.unexpected_keys)})"
                )
            else:
                print("[clip] INFO: No 'visual_projection_state_dict' — skipping.")

            # ── Class mapping (informational) ────────────────────────────────
            class_to_idx = checkpoint.get("class_to_idx", {})
            if class_to_idx:
                print(f"[clip] Checkpoint classes: {list(class_to_idx.keys())}")

            print("[clip] Fine-tuned weights applied successfully.")

        except Exception as exc:
            print(f"[clip] WARNING: Failed to apply fine-tuned weights: {exc}")
            print("[clip] Falling back to base CLIP weights.")

    def _norm(self, emb: torch.Tensor) -> torch.Tensor:
        return emb / emb.norm(dim=-1, keepdim=True)

    def _get_image_features(self, inputs) -> torch.Tensor:
        with torch.no_grad():
            out = self.model.get_image_features(**inputs)
            if hasattr(out, "pooler_output"):
                out = out.pooler_output
        return self._norm(out).cpu()

    def _get_text_features(self, inputs) -> torch.Tensor:
        with torch.no_grad():
            out = self.model.get_text_features(**inputs)
            if hasattr(out, "pooler_output"):
                out = out.pooler_output
        return self._norm(out).cpu()

    # ── Basic encoding ───────────────────────────────────────────────────────

    def encode_image(self, path: str) -> torch.Tensor:
        """Encode single image file -> (1, D) normalized."""
        img = Image.open(path).convert("RGB")
        return self.encode_image_pil(img)

    def encode_image_pil(self, image: Image.Image) -> torch.Tensor:
        """Encode single PIL image -> (1, D) normalized."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        return self._get_image_features(inputs)

    def encode_image_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """Encode batch of PIL images -> (N, D) normalized."""
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        return self._get_image_features(inputs)

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode single text -> (1, D) normalized."""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        return self._get_text_features(inputs)

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode multiple texts -> (N, D) normalized."""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        return self._get_text_features(inputs)

    # ── Multi-scale encoding ─────────────────────────────────────────────────

    def encode_image_multiscale(
        self, image: Image.Image, scales: List[int] = [224, 384, 512]
    ) -> torch.Tensor:
        """
        Encode image at multiple resolutions, return the mean embedding.

        Captures both fine details (small scale) and global context (large scale).
        """
        embs = []
        for size in scales:
            resized = image.resize((size, size), Image.LANCZOS)
            embs.append(self.encode_image_pil(resized))
        mean_emb = torch.mean(torch.cat(embs, dim=0), dim=0, keepdim=True)
        return self._norm(mean_emb)

    # ── Region encoding ──────────────────────────────────────────────────────

    def encode_regions(
        self, image: Image.Image, boxes: List[Tuple[int, int, int, int]],
        batch_size: int = 16
    ) -> torch.Tensor:
        """
        Encode cropped regions from an image.

        Args:
            image: PIL Image.
            boxes: List of (x, y, w, h).
            batch_size: Inference batch size.

        Returns:
            (N, D) normalized embeddings.
        """
        crops = []
        for (x, y, w, h) in boxes:
            crop = image.crop((x, y, x + w, y + h))
            crops.append(crop)

        if not crops:
            return torch.zeros(0, 512)

        all_embs = []
        for i in range(0, len(crops), batch_size):
            batch = crops[i:i + batch_size]
            embs = self.encode_image_batch(batch)
            all_embs.append(embs)
        return torch.cat(all_embs, dim=0)

    # ── Support augmentation ─────────────────────────────────────────────────

    def augment_and_encode(
        self, image: Image.Image, n_augments: int = 8
    ) -> torch.Tensor:
        """
        Create augmented views of a support image and encode all of them.
        Returns (n_augments+1, D) embeddings (original + augmented).

        Augmentations: horizontal flip, random crops, brightness/contrast shifts.
        """
        views = [image]  # original

        w, h = image.size

        # Horizontal flip
        views.append(image.transpose(Image.FLIP_LEFT_RIGHT))

        # Center crop (80%)
        cw, ch = int(w * 0.1), int(h * 0.1)
        views.append(image.crop((cw, ch, w - cw, h - ch)).resize((w, h), Image.LANCZOS))

        # 4 corner crops (70%)
        crop_w, crop_h = int(w * 0.7), int(h * 0.7)
        for (cx, cy) in [(0, 0), (w - crop_w, 0), (0, h - crop_h), (w - crop_w, h - crop_h)]:
            views.append(image.crop((cx, cy, cx + crop_w, cy + crop_h)).resize((w, h), Image.LANCZOS))

        # Brightness variation
        enhancer = ImageEnhance.Brightness(image)
        views.append(enhancer.enhance(1.3))

        # Contrast variation
        enhancer = ImageEnhance.Contrast(image)
        views.append(enhancer.enhance(1.3))

        # Limit to requested count
        views = views[:n_augments + 1]

        return self.encode_image_batch(views)
