#!/usr/bin/env python3
"""
Class-incremental learning on ModelNet40 with LwF, following PyCIL's loss.
Tasks introduce 4 classes at a time (10 tasks total).
"""

import argparse
import copy
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

POINTNET_ROOT = os.path.join(os.path.dirname(__file__), "PointNet")
if POINTNET_ROOT not in sys.path:
    sys.path.insert(0, POINTNET_ROOT)

from pointnet.dataset import ModelNetDataset
from pointnet.model import PointNetCls

NUM_POINTS = 1024
NUM_CLASSES = 40

BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 5e-4
TEMPERATURE = 2.0
LWF_LAMBDA = 3.0  # Matches PyCIL default for LwF.

MODELNET_DEFAULT_ROOT = os.path.join(POINTNET_ROOT, "data")
MODELNET_ID_PATH = os.path.join(POINTNET_ROOT, "misc", "modelnet_id.txt")



def set_deterministic():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)


def load_pointnet_class_map() -> Dict[int, str]:
    if not os.path.isfile(MODELNET_ID_PATH):
        raise RuntimeError(f"Missing class map at {MODELNET_ID_PATH}")
    mapping = {}
    with open(MODELNET_ID_PATH, "r", encoding="utf-8") as f:
        for line in f:
            name, idx_str = line.strip().split()
            mapping[int(idx_str)] = name
    return mapping


CLASS_IDX_TO_NAME = load_pointnet_class_map()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continual LwF training on ModelNet40 via PointNet's dataset."
    )
    parser.add_argument(
        "--modelnet-root",
        type=str,
        default=MODELNET_DEFAULT_ROOT,
        help="Root directory containing PointNet's train.txt/test.txt and ply files.",
    )
    return parser.parse_args()


def build_modelnet_dataset(
    root: str,
    split: str,
    num_points: int,
    class_indices: Optional[List[int]],
    augment: bool,
):
    dataset = ModelNetDataset(
        root=root,
        npoints=num_points,
        split=split,
        data_augmentation=augment,
    )
    if not class_indices:
        return dataset
    allowed = {CLASS_IDX_TO_NAME[idx] for idx in class_indices}
    dataset.fns = [fn for fn in dataset.fns if fn.split("/")[0] in allowed]
    return dataset


def kd_loss(pred_log_probs: torch.Tensor, old_log_probs: torch.Tensor, temperature: float) -> torch.Tensor:
    scaled_pred = pred_log_probs / temperature
    scaled_old = old_log_probs / temperature
    soft_old = torch.exp(scaled_old)
    return (soft_old * (scaled_old - scaled_pred)).sum(dim=1).mean()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    old_classes: List[int],
    new_classes: List[int],
) -> Tuple[float, float, float, Dict[int, float]]:
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for points, labels in loader:
            points = points.transpose(1, 2).to(next(model.parameters()).device)
            labels = labels.to(points.device).squeeze(-1)
            log_probs, _, _ = model(points)
            preds = log_probs.argmax(dim=1)
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
    labels = torch.cat(all_labels)
    preds = torch.cat(all_preds)
    overall = (preds == labels).float().mean().item()

    def acc_for(classes: List[int]) -> float:
        if not classes:
            return 0.0
        class_tensor = torch.tensor(classes)
        mask = torch.isin(labels, class_tensor)
        if mask.sum() == 0:
            return 0.0
        return (preds[mask] == labels[mask]).float().mean().item()

    old_acc = acc_for(old_classes)
    new_acc = acc_for(new_classes)

    per_class = {}
    for c in sorted(set(old_classes + new_classes)):
        mask = labels == c
        if mask.sum() == 0:
            per_class[c] = 0.0
        else:
            per_class[c] = (preds[mask] == labels[mask]).float().mean().item()
    return overall, old_acc, new_acc, per_class


def train_task(
    model: nn.Module,
    loader: DataLoader,
    old_model: Optional[nn.Module],
    known_classes: int,
    total_classes: int,
    device: torch.device,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    model.train()
    for epoch in range(EPOCHS):
        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for points, labels in progress:
            points = points.transpose(1, 2).to(device)
            labels = labels.to(device).squeeze(-1)
            log_probs, _, _ = model(points)
            if old_model is None:
                loss = F.nll_loss(log_probs[:, :total_classes], labels)
            else:
                fake_targets = labels - known_classes
                loss_clf = F.nll_loss(
                    log_probs[:, known_classes:total_classes], fake_targets
                )
                with torch.no_grad():
                    old_log_probs, _, _ = old_model(points)
                loss_kd = kd_loss(
                    log_probs[:, :known_classes],
                    old_log_probs[:, :known_classes],
                    TEMPERATURE,
                )
                loss = LWF_LAMBDA * loss_kd + loss_clf
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def main():
    args = parse_args()
    set_deterministic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelnet_root = args.modelnet_root

    tasks = [list(range(i, i + 4)) for i in range(0, NUM_CLASSES, 4)]

    model = PointNetCls(k=NUM_CLASSES, feature_transform=False).to(device)
    max_class_acc: Dict[int, float] = {}

    for task_id, task_classes in enumerate(tasks, start=1):
        known_classes = sum(len(t) for t in tasks[: task_id - 1])
        total_classes = known_classes + len(task_classes)

        train_dataset = build_modelnet_dataset(
            root=modelnet_root,
            split="train",
            num_points=NUM_POINTS,
            class_indices=task_classes,
            augment=True,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True
        )

        old_model = None
        if task_id > 1:
            old_model = copy.deepcopy(model).to(device)
            old_model.eval()
            for p in old_model.parameters():
                p.requires_grad = False

        train_task(model, train_loader, old_model, known_classes, total_classes, device)

        seen_classes = sum(tasks[:task_id], [])
        test_dataset = build_modelnet_dataset(
            root=modelnet_root,
            split="test",
            num_points=NUM_POINTS,
            class_indices=seen_classes,
            augment=False,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )

        old_classes = sum(tasks[: task_id - 1], [])
        new_classes = task_classes
        overall, old_acc, new_acc, per_class = evaluate(
            model, test_loader, old_classes, new_classes
        )

        print(
            f"Task {task_id} | Overall Acc {overall*100:.2f}% | "
            f"Old Acc {old_acc*100:.2f}% | New Acc {new_acc*100:.2f}%"
        )

        forgetting_values = []
        for c, acc in per_class.items():
            if c in max_class_acc:
                forgetting_values.append(max_class_acc[c] - acc)
            max_class_acc[c] = max(max_class_acc.get(c, 0.0), acc)
        if forgetting_values:
            avg_forgetting = sum(forgetting_values) / len(forgetting_values)
        else:
            avg_forgetting = 0.0
        print(f"Forgetting (avg over seen classes): {avg_forgetting*100:.2f}%")


if __name__ == "__main__":
    main()
