"""
End-to-end NSVR (base) on CLEVR (val split):

1) Load CNN checkpoint (auto-pick best/latest .pt)
2) Run batched inference on object crops (from pixel_coords)
3) Build PREDICTED scene graphs (nodes = predicted attrs, edges = GT relations)
4) Execute CLEVR programs over predicted graphs (symbolic executor)
5) Report VQA accuracy and save graphs to data/processed/

Run:
    python src/pipeline/run_end2end.py
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict
import os, sys, json

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

# Make "src/" importable when running this file directly
SRC_DIR = Path(__file__).resolve().parents[1]      # .../src
PROJ_ROOT = SRC_DIR.parent                         # repo root
sys.path.append(str(SRC_DIR))

# Local imports
from data.clevr_dataset import ClevrObjectAttributes, VOCAB
from models.resnet_multitask import ResNetMultiHead
from reasoning.reason_clevr import CLEVRExecutor   # base (hard) executor

def P(x):
    return x if isinstance(x, Path) else Path(x)

# ---------------- Defaults (relative to repo root) ----------------
CLEVR_ROOT = P(r"C:\Users\Sriman Rakshan N\Documents\Amrita\Project_Sem_V\data\raw\clevr\CLEVR_v1.0")
OUT_DIR    = P(r"C:\Users\Sriman Rakshan N\Documents\Amrita\Project_Sem_V\data\processed")
CKPT_DIR   = P(os.environ.get("CKPT_DIR", str(PROJ_ROOT / "checkpoints")))  # where train_perception.py saved checkpoints
SPLIT      = "val"
IMG_SIZE   = 128
BATCH_SIZE = 128
NUM_WORKERS = 4
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Utils ----------------
def invert_vocab(v: Dict[str, int]) -> Dict[int, str]:
    return {i: s for s, i in v.items()}

INV = {
    "color":    invert_vocab(VOCAB["color"]),
    "shape":    invert_vocab(VOCAB["shape"]),
    "material": invert_vocab(VOCAB["material"]),
    "size":     invert_vocab(VOCAB["size"]),
}

def collate_with_meta(batch):
    """Return (images tensor, labels dict, metas list) so we can keep per-sample metadata."""
    import torch as _torch
    images = _torch.stack([b["image"] for b in batch], dim=0)
    labels = {
        "color": _torch.tensor([b["labels"]["color_id"] for b in batch], dtype=_torch.long),
        "shape": _torch.tensor([b["labels"]["shape_id"] for b in batch], dtype=_torch.long),
        "material": _torch.tensor([b["labels"]["material_id"] for b in batch], dtype=_torch.long),
        "size": _torch.tensor([b["labels"]["size_id"] for b in batch], dtype=_torch.long),
    }
    metas = [b["meta"] for b in batch]
    return images, labels, metas

def get_checkpoint_path() -> Path:
    """Pick best_model.pt if present; else latest .pt in CKPT_DIR; else error."""
    best = CKPT_DIR / "best_model.pt"
    if best.exists():
        return best
    pts = sorted(CKPT_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if pts:
        return pts[0]
    raise FileNotFoundError(f"No checkpoints found in {CKPT_DIR}. Train the model first.")

def load_scenes(clevr_root: str | Path, split: str) -> Dict[int, Dict[str, Any]]:
    clevr_root = Path(clevr_root)  # ensure Path
    p = clevr_root / "scenes" / f"CLEVR_{split}_scenes.json"
    with open(p, "r") as f:
        scenes = json.load(f)["scenes"]
    return {s["image_index"]: s for s in scenes}

def save_jsonl(records: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

def load_questions(clevr_root: str | Path, split: str) -> List[Dict[str, Any]]:
    clevr_root = Path(clevr_root)  # ensure Path
    q = clevr_root / "questions" / f"CLEVR_{split}_questions.json"
    with open(q, "r") as f:
        return json.load(f)["questions"]

def build_pred_graphs(
    per_image: Dict[int, List[Dict[str, Any]]],
    scenes_by_id: Dict[int, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Construct predicted graphs using predicted attrs + GT relations/positions."""
    graphs = []
    mapping = {"left":"left_of", "right":"right_of", "front":"front_of", "behind":"behind"}
    for image_id, objs in per_image.items():
        scene = scenes_by_id[image_id]
        nodes = []
        for rec in sorted(objs, key=lambda x: x["obj_index"]):
            gt_obj = scene["objects"][rec["obj_index"]]
            nodes.append({
                "id": rec["obj_index"],
                "color": rec["pred"]["color"],
                "shape": rec["pred"]["shape"],
                "size": rec["pred"]["size"],
                "material": rec["pred"]["material"],
                "position": gt_obj["3d_coords"],
                "probs": rec.get("probs", None),   # <<< NEW: pass attribute probabilities
            })
        edges = []
        rels = scene.get("relationships", {})
        for rkey, rname in mapping.items():
            lists = rels.get(rkey, [])
            for src_idx, dsts in enumerate(lists):
                for d in dsts:
                    edges.append({"src": src_idx, "rel": rname, "dst": d})
        graphs.append({
            "image_id": image_id,
            "image_filename": scene["image_filename"],
            "nodes": nodes,
            "edges": edges,
        })
    return graphs

def evaluate_vqa(graphs_by_id: Dict[int, Dict[str,Any]], questions: List[Dict[str,Any]], preview: int = 5):
    total, correct = 0, 0
    for q in questions:
        img_id = q["image_index"]
        if img_id not in graphs_by_id:
            continue
        trace = []
        ex = CLEVRExecutor(graphs_by_id[img_id])
        try:
            pred = ex.execute(q["program"], trace=trace)
            pred_norm = "yes" if isinstance(pred, bool) and pred else ("no" if isinstance(pred, bool) else str(pred))
            gt_norm = str(q.get("answer"))
            ok = (pred_norm == gt_norm)
        except Exception:
            ok = False
        total += 1
        correct += int(ok)
        if preview > 0:
            print(f"[preview] img={img_id} ok={ok}  Q: {q['question']}")
            preview -= 1
    acc = correct / max(total, 1)
    print(f"\nVQA over predicted graphs → Total: {total} | Correct: {correct} | Acc: {acc:.4f}")

# ---------------- Main ----------------
def main():
    print("=== NSVR End-to-End (Base) ===")
    print("Project root   :", PROJ_ROOT)
    print("CLEVR root     :", CLEVR_ROOT)
    print("Output dir     :", OUT_DIR)
    print("Checkpoint dir :", CKPT_DIR)
    device = torch.device(DEVICE)
    print("Device         :", device)

    # 1) Dataset & loader (val split)
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_ds = ClevrObjectAttributes(str(CLEVR_ROOT), split=SPLIT, img_size=IMG_SIZE, transform=transform)
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_with_meta
    )
    print(f"Val objects: {len(val_ds)}")

    # 2) Load model checkpoint
    ckpt_path = get_checkpoint_path()
    print("Using checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    model = ResNetMultiHead(pretrained=False, embedding_dim=1024).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 3) Batched inference → per-image predictions (WITH probabilities)
    per_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    with torch.no_grad():
        for imgs, _, metas in tqdm(val_loader, desc="Perception inference (val)"):
            imgs = imgs.to(device)
            out = model(imgs)

            # --- NEW: per-head softmax probabilities (on CPU) ---
            p_heads = {h: torch.softmax(out[h], dim=1).cpu() for h in out.keys()}

            # per-sample argmax + probs
            for i, m in enumerate(metas):
                pred = {
                    "color":    INV["color"][out["color"][i].argmax().item()],
                    "shape":    INV["shape"][out["shape"][i].argmax().item()],
                    "material": INV["material"][out["material"][i].argmax().item()],
                    "size":     INV["size"][out["size"][i].argmax().item()],
                }
                # --- NEW: map probs -> {label: prob} for each head
                label_probs = {
                    head: {INV[head][j]: float(p_heads[head][i, j]) for j in range(p_heads[head].shape[1])}
                    for head in ("color", "shape", "material", "size")
                }
                per_image[m["image_id"]].append({
                    "obj_index": m["obj_index"],
                    "pred": pred,
                    "probs": label_probs,  # <<< NEW
                })

    # 4) Build predicted graphs and save
    scenes_by_id = load_scenes(CLEVR_ROOT, SPLIT)
    pred_graphs = build_pred_graphs(per_image, scenes_by_id)
    out_graphs_path = OUT_DIR / f"pred_scene_graphs_{SPLIT}.jsonl"
    save_jsonl(pred_graphs, out_graphs_path)
    print(f"Saved predicted graphs → {out_graphs_path}")

    # 5) Reasoning over predicted graphs
    graphs_by_id = {g["image_id"]: g for g in pred_graphs}
    questions = load_questions(CLEVR_ROOT, SPLIT)
    evaluate_vqa(graphs_by_id, questions, preview=5)

if __name__ == "__main__":
    main()
