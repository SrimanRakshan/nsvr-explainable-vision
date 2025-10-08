from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import os, sys, json

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

# sklearn is optional; we degrade gracefully if missing
try:
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    SK_OK = True
except Exception:
    SK_OK = False

# ----- repo paths -----
SRC_DIR   = Path(__file__).resolve().parents[1]
PROJ_ROOT = SRC_DIR.parent
sys.path.append(str(SRC_DIR))

from data.clevr_dataset import ClevrObjectAttributes, VOCAB
from models.resnet_multitask import ResNetMultiHead

# ----- config -----
CLEVR_ROOT = r"C:\Users\Sriman Rakshan N\Documents\Amrita\Project_Sem_V\data\raw\clevr\CLEVR_v1.0"
CKPT_DIR   = Path(os.environ.get("CKPT_DIR",   PROJ_ROOT / "checkpoints"))
OUT_DIR    = PROJ_ROOT / "data" / "results"
SPLIT      = "val"
IMG_SIZE   = 128
BATCH_SIZE = 128
NUM_WORKERS = 4
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

def invert_vocab(v: Dict[str,int]) -> Dict[int,str]:
    return {i:s for s,i in v.items()}

INV = {k: invert_vocab(VOCAB[k]) for k in VOCAB.keys()}

def labels_in_order(head: str) -> List[int]:
    # get ids sorted by index 0..n-1
    return [i for i,_ in sorted(INV[head].items(), key=lambda x: x[0])]

def names_in_order(head: str) -> List[str]:
    return [INV[head][i] for i in labels_in_order(head)]

def collate_with_meta(batch):
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
    best = CKPT_DIR / "best_model.pt"
    if best.exists(): return best
    pts = sorted(CKPT_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if pts: return pts[0]
    raise FileNotFoundError(f"No checkpoints found in {CKPT_DIR}")

def main():
    print("=== Evaluate Perception (CNN) ===")
    print("CLEVR:", CLEVR_ROOT)
    print("CKPT :", CKPT_DIR)
    print("OUT  :", OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(DEVICE)
    print("Device:", device)

    # data
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_ds = ClevrObjectAttributes(str(CLEVR_ROOT), split=SPLIT, img_size=IMG_SIZE, transform=transform)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, collate_fn=collate_with_meta)
    print("Val objects:", len(val_ds))

    # model
    ckpt_path = get_checkpoint_path()
    print("Using checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    model = ResNetMultiHead(pretrained=False, embedding_dim=1024).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # collect predictions/targets per head
    y_true: Dict[str, List[int]] = {h: [] for h in VOCAB.keys()}
    y_pred: Dict[str, List[int]] = {h: [] for h in VOCAB.keys()}

    with torch.no_grad():
        for imgs, labels, _ in tqdm(val_loader, desc="Eval (val)"):
            imgs = imgs.to(device)
            out = model(imgs)
            for head in VOCAB.keys():
                y_true[head].extend(labels[head].tolist())
                y_pred[head].extend(out[head].argmax(dim=1).cpu().tolist())

    # build summary
    summary: Dict[str, Any] = {"split": SPLIT, "counts": {h: len(y_true[h]) for h in y_true}}
    for head in VOCAB.keys():
        acc = sum(int(a==b) for a,b in zip(y_true[head], y_pred[head])) / max(1, len(y_true[head]))
        summary[head] = {"accuracy": acc}

        if SK_OK:
            labels_ord = labels_in_order(head)
            names_ord  = names_in_order(head)
            cm = confusion_matrix(y_true[head], y_pred[head], labels=labels_ord)
            report = classification_report(y_true[head], y_pred[head], labels=labels_ord,
                                           target_names=names_ord, zero_division=0, output_dict=True)
            summary[head]["classification_report"] = report
            # save CM to CSV
            import csv
            cm_path = OUT_DIR / f"confusion_{head}.csv"
            with open(cm_path, "w", newline="") as f:
                w = csv.writer(f); w.writerow([""] + names_ord)
                for i, row in enumerate(cm):
                    w.writerow([names_ord[i]] + row.tolist())
        else:
            summary[head]["note"] = "Install scikit-learn for confusion matrices and detailed report."

    # save summary
    with open(OUT_DIR / "perception_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved:", OUT_DIR / "perception_metrics.json")
    if SK_OK:
        print("Saved confusion matrices to:", OUT_DIR)

if __name__ == "__main__":
    main()
