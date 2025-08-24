# resnet_multitask.py
from __future__ import annotations
import torch
import torch.nn as nn
from torchvision import models

class ResNetMultiHead(nn.Module):
    """
    ResNet50 backbone with 4 classification heads:
      - color (8)
      - shape (3)
      - material (2)
      - size (2)
    Each head is a small MLP on top of backbone features.
    """
    def __init__(self, pretrained: bool = True, embedding_dim: int = 1024):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)
        # remove final fc
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # output: (B, 2048, 1, 1)
        feat_dim = 2048

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        # heads
        self.color_head = nn.Linear(embedding_dim, 8)
        self.shape_head = nn.Linear(embedding_dim, 3)
        self.material_head = nn.Linear(embedding_dim, 2)
        self.size_head = nn.Linear(embedding_dim, 2)

        # init heads
        for m in (self.color_head, self.shape_head, self.material_head, self.size_head):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> dict:
        # x: (B, 3, H, W)
        f = self.backbone(x)           # (B, 2048, 1, 1)
        z = self.proj(f)               # (B, embedding_dim)
        return {
            "color": self.color_head(z),
            "shape": self.shape_head(z),
            "material": self.material_head(z),
            "size": self.size_head(z),
        }
