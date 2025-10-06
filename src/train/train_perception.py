# train_perception.py
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
from typing import Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision import transforms as T

from data.clevr_dataset import ClevrObjectAttributes, VOCAB  # assumes clevr_dataset.py in same folder
from models.resnet_multitask import ResNetMultiHead

# ---------- utilities ----------
def seed_everything(seed: int = 1337):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = {
        "color": torch.tensor([b["labels"]["color_id"] for b in batch], dtype=torch.long),
        "shape": torch.tensor([b["labels"]["shape_id"] for b in batch], dtype=torch.long),
        "material": torch.tensor([b["labels"]["material_id"] for b in batch], dtype=torch.long),
        "size": torch.tensor([b["labels"]["size_id"] for b in batch], dtype=torch.long),
    }
    return images, labels

def compute_accuracy(preds: torch.Tensor, target: torch.Tensor) -> float:
    pred = preds.argmax(dim=1)
    correct = (pred == target).float().sum().item()
    return correct / target.size(0)

# ---------- training / eval ----------
def train_epoch(model, loader, opt, device, loss_fns, lambda_weights):
    model.train()
    total_loss = 0.0
    metrics = {"color":0,"shape":0,"material":0,"size":0}
    seen = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        lbls = {k: v.to(device) for k,v in labels.items()}
        out = model(imgs)
        loss = 0.0
        for k, loss_fn in loss_fns.items():
            l = loss_fn(out[k], lbls[k]) * lambda_weights[k]
            loss = loss + l
        opt.zero_grad()
        loss.backward()
        opt.step()
        b = imgs.size(0)
        total_loss += loss.item() * b
        seen += b
        # accumulate accuracies
        for k in metrics:
            metrics[k] += compute_accuracy(out[k].detach().cpu(), lbls[k].detach().cpu()) * b
    avg_loss = total_loss / seen
    accs = {k: metrics[k] / seen for k in metrics}
    return avg_loss, accs

@torch.no_grad()
def eval_epoch(model, loader, device, loss_fns, lambda_weights):
    model.eval()
    total_loss = 0.0
    metrics = {"color":0,"shape":0,"material":0,"size":0}
    seen = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        lbls = {k: v.to(device) for k,v in labels.items()}
        out = model(imgs)
        loss = 0.0
        for k, loss_fn in loss_fns.items():
            l = loss_fn(out[k], lbls[k]) * lambda_weights[k]
            loss = loss + l
        b = imgs.size(0)
        total_loss += loss.item() * b
        seen += b
        for k in metrics:
            metrics[k] += compute_accuracy(out[k].detach().cpu(), lbls[k].detach().cpu()) * b
    avg_loss = total_loss / seen
    accs = {k: metrics[k] / seen for k in metrics}
    return avg_loss, accs

# ---------- main ----------
def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # dataset & loaders
    transform = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    train_ds = ClevrObjectAttributes(args.data_root, split="train", img_size=args.img_size, transform=transform)
    val_ds = ClevrObjectAttributes(args.data_root, split="val", img_size=args.img_size, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    print("Dataset sizes:", len(train_ds), "train samples;", len(val_ds), "val samples")

    # model
    model = ResNetMultiHead(pretrained=args.pretrained, embedding_dim=args.embedding_dim).to(device)

    # loss & optimizer
    loss_fns = {
        "color": nn.CrossEntropyLoss(),
        "shape": nn.CrossEntropyLoss(),
        "material": nn.CrossEntropyLoss(),
        "size": nn.CrossEntropyLoss(),
    }
    # simple equal weighting (can tune later)
    lambda_weights = {"color":1.0, "shape":1.0, "material":1.0, "size":1.0}

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_acc = 0.0
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        train_loss, train_accs = train_epoch(model, train_loader, optimizer, device, loss_fns, lambda_weights)
        val_loss, val_accs = eval_epoch(model, val_loader, device, loss_fns, lambda_weights)
        scheduler.step()

        # average val acc across heads as simple scalar
        avg_val_acc = sum(val_accs.values()) / len(val_accs)
        print(f"Epoch {epoch}/{args.epochs} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_accs {val_accs} | avg_val_acc {avg_val_acc:.4f}")

        # checkpoint best
        ckpt_path = Path(args.save_dir) / f"model_epoch{epoch}.pt"
        torch.save({"epoch":epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}, ckpt_path)

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_path = Path(args.save_dir) / "best_model.pt"
            torch.save({"epoch":epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "vocab": VOCAB}, best_path)
            print("Saved new best:", best_path)

    # final vocab dump
    with open(Path(args.save_dir) / "vocab.json", "w") as f:
        json.dump(VOCAB, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="/home/Zoro/Documents/data/raw/clevr/CLEVR_v1.0/", help="CLEVR_v1.0 directory")
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--pretrained", action="store_true", help="use imagenet pretrained backbone")
    parser.add_argument("--embedding-dim", type=int, default=1024)
    args = parser.parse_args()
    main(args)
