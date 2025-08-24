# Step 3.1 — CLEVR object-attribute dataset (per-object crops)
# Save as: clevr_dataset.py

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple, List
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

# ---- Attribute vocab (CLEVR canonical) ----
COLORS = ["gray","red","blue","green","brown","purple","cyan","yellow"]
SHAPES = ["cube","sphere","cylinder"]
MATERIALS = ["rubber","metal"]
SIZES = ["small","large"]

VOCAB = {
    "color": {c: i for i, c in enumerate(COLORS)},
    "shape": {s: i for i, s in enumerate(SHAPES)},
    "material": {m: i for i, m in enumerate(MATERIALS)},
    "size": {s: i for i, s in enumerate(SIZES)},
}

def _default_transform(img_size: int = 128):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

def _compute_crop_box(cx: float, cy: float, size: str, W: int, H: int) -> Tuple[int,int,int,int]:
    """
    Build a square crop around (cx, cy) in pixel space.
    Heuristic side length based on CLEVR 'size' and image resolution.
    """
    # Heuristic scale (CLEVR images are 320x240)
    base = max(W, H)
    if size == "small":
        side = int(0.22 * base)  # ~70px for 320
    else:  # large
        side = int(0.32 * base)  # ~100px for 320

    x1 = int(cx - side // 2)
    y1 = int(cy - side // 2)
    x2 = x1 + side
    y2 = y1 + side

    # clamp to image bounds
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W, x2); y2 = min(H, y2)
    return x1, y1, x2, y2

class ClevrObjectAttributes(Dataset):
    """
    Per-object dataset for CLEVR:
      - Each item is one object crop + attribute labels.
      - Uses scenes JSON 'pixel_coords' as center; crops are heuristic.
    """
    def __init__(
        self,
        clevr_root: str | Path,
        split: str = "train",
        img_size: int = 128,
        transform = None,
    ):
        """
        clevr_root -> path to CLEVR_v1.0 (folder that contains images/, scenes/, questions/)
        split      -> 'train' | 'val' (test has no scene JSON in CLEVR)
        """
        self.root = Path(clevr_root)
        assert (self.root / "images").exists(), f"images/ not found under {self.root}"
        assert split in ("train", "val"), "split must be 'train' or 'val'"

        self.split = split
        scenes_json = self.root / "scenes" / f"CLEVR_{split}_scenes.json"
        assert scenes_json.exists(), f"Scenes file not found: {scenes_json}"

        with open(scenes_json, "r") as f:
            data = json.load(f)
        self.scenes = data["scenes"]

        self.img_dir = self.root / "images" / split
        self.transform = transform or _default_transform(img_size)

        # Build index: list of (image_filename, image_id, obj_dict, obj_index)
        self.index: List[Dict[str, Any]] = []
        for scene in self.scenes:
            img_idx = scene["image_index"]
            image_filename = scene["image_filename"]  # e.g., CLEVR_train_000000.png
            for j, obj in enumerate(scene["objects"]):
                # Ensure required fields exist
                pc = obj.get("pixel_coords", None)  # [cx, cy, depth]
                if not pc:
                    # Fallback: skip objects without pixel coords (rare)
                    continue
                entry = {
                    "image_id": img_idx,
                    "image_filename": image_filename,
                    "obj": {
                        "color": obj["color"],
                        "shape": obj["shape"],
                        "material": obj["material"],
                        "size": obj["size"],
                        "pixel_coords": pc,  # [cx, cy, depth]
                    },
                    "obj_index": j,
                }
                self.index.append(entry)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        rec = self.index[i]
        img_path = self.img_dir / rec["image_filename"]
        img = Image.open(img_path).convert("RGB")

        W, H = img.size
        cx, cy, _ = rec["obj"]["pixel_coords"]
        size = rec["obj"]["size"]
        x1, y1, x2, y2 = _compute_crop_box(cx, cy, size, W, H)
        crop = img.crop((x1, y1, x2, y2))

        if self.transform:
            crop = self.transform(crop)

        labels = {
            "color_id": VOCAB["color"][rec["obj"]["color"]],
            "shape_id": VOCAB["shape"][rec["obj"]["shape"]],
            "material_id": VOCAB["material"][rec["obj"]["material"]],
            "size_id": VOCAB["size"][rec["obj"]["size"]],
        }
        meta = {
            "image_id": rec["image_id"],
            "obj_index": rec["obj_index"],
            "center": (cx, cy),
            "bbox": (x1, y1, x2, y2),
            "image_path": str(img_path),
        }
        return {"image": crop, "labels": labels, "meta": meta}

# ---- quick sanity run ----
if __name__ == "__main__":
    # Example:
    # ROOT = "/home/you/data/raw/CLEVR_v1.0"
    ROOT = "/home/Zoro/Documents/data/raw/clevr/CLEVR_v1.0/"
    ds = ClevrObjectAttributes(ROOT, split="train", img_size=128)
    print("Total objects (train):", len(ds))
    sample = ds[0]
    print("Sample tensor shape:", sample["image"].shape)
    print("Labels:", sample["labels"])
    print("Meta:", sample["meta"])
