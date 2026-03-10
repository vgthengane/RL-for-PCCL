#!/usr/bin/env python3

import copy
import os
import random
import urllib.request
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


DATA_URL = "http://modelnet.cs.princeton.edu/ModelNet40.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
NUM_POINTS = 1024
NUM_CLASSES = 40

BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 5e-4
LWF_LAMBDA = 1.0


def set_deterministic():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)


def download_modelnet40(root: str):
    os.makedirs(root, exist_ok=True)
    modelnet_dir = os.path.join(root, "ModelNet40")
    if os.path.isdir(modelnet_dir):
        return modelnet_dir
    zip_path = os.path.join(root, "ModelNet40.zip")
    if not os.path.isfile(zip_path):
        print("Downloading ModelNet40...")
        urllib.request.urlretrieve(DATA_URL, zip_path)
    print("Extracting ModelNet40...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(root)
    return modelnet_dir


def load_class_names(modelnet_dir: str) -> List[str]:
    shape_names = os.path.join(modelnet_dir, "modelnet40_shape_names.txt")
    if os.path.isfile(shape_names):
        with open(shape_names, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return sorted(
        [
            d
            for d in os.listdir(modelnet_dir)
            if os.path.isdir(os.path.join(modelnet_dir, d))
        ]
    )


def read_off(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        if header.startswith("OFF"):
            if header != "OFF":
                header = header[3:]
        else:
            raise ValueError(f"Invalid OFF header in {path}")
        if header:
            counts = header.split()
        else:
            counts = f.readline().strip().split()
        n_verts, n_faces = int(counts[0]), int(counts[1])
        verts = []
        for _ in range(n_verts):
            verts.append([float(v) for v in f.readline().strip().split()])
        verts = np.array(verts, dtype=np.float32)
        faces = []
        for _ in range(n_faces):
            parts = [int(p) for p in f.readline().strip().split()]
            k = parts[0]
            idx = parts[1:]
            if k == 3:
                faces.append(idx)
            elif k > 3:
                for i in range(1, k - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
        faces = np.array(faces, dtype=np.int64)
    return verts, faces


def sample_points_from_mesh(verts: np.ndarray, faces: np.ndarray, num_points: int) -> np.ndarray:
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    face_probs = face_areas / (face_areas.sum() + 1e-12)
    face_indices = np.random.choice(len(faces), size=num_points, p=face_probs)
    v0 = v0[face_indices]
    v1 = v1[face_indices]
    v2 = v2[face_indices]
    u = np.random.rand(num_points, 1)
    v = np.random.rand(num_points, 1)
    sqrt_u = np.sqrt(u)
    points = (1 - sqrt_u) * v0 + sqrt_u * (1 - v) * v1 + sqrt_u * v * v2
    return points.astype(np.float32)


def normalize_point_cloud(points: np.ndarray) -> np.ndarray:
    centroid = points.mean(axis=0, keepdims=True)
    points = points - centroid
    scale = np.linalg.norm(points, axis=1).max()
    if scale > 0:
        points = points / scale
    return points.astype(np.float32)


def random_rotate_z(points: np.ndarray) -> np.ndarray:
    theta = np.random.uniform(0.0, 2.0 * np.pi)
    cosval = np.cos(theta)
    sinval = np.sin(theta)
    rotation = np.array(
        [[cosval, -sinval, 0.0], [sinval, cosval, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return points @ rotation.T


def random_scale(points: np.ndarray, scale_low: float = 0.8, scale_high: float = 1.25) -> np.ndarray:
    scale = np.random.uniform(scale_low, scale_high)
    return points * scale


def random_shift(points: np.ndarray, shift_range: float = 0.1) -> np.ndarray:
    shift = np.random.uniform(-shift_range, shift_range, size=(1, 3)).astype(np.float32)
    return points + shift


def jitter_points(points: np.ndarray, sigma: float = 0.01, clip: float = 0.05) -> np.ndarray:
    noise = np.clip(sigma * np.random.randn(*points.shape), -clip, clip).astype(np.float32)
    return points + noise


def random_point_dropout(points: np.ndarray, max_dropout_ratio: float = 0.875) -> np.ndarray:
    dropout_ratio = np.random.uniform(0.0, max_dropout_ratio)
    drop_idx = np.where(np.random.rand(points.shape[0]) <= dropout_ratio)[0]
    if drop_idx.size > 0:
        points[drop_idx, :] = points[0, :]
    return points


class ModelNet40(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        num_points: int = NUM_POINTS,
        class_indices: Optional[List[int]] = None,
    ):
        self.modelnet_dir = download_modelnet40(root)
        self.class_names = load_class_names(self.modelnet_dir)
        if len(self.class_names) != NUM_CLASSES:
            raise RuntimeError("Expected 40 classes in ModelNet40.")
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.num_points = num_points
        self.augment = split == "train"
        self.files: List[Tuple[str, int]] = []
        allowed = set(class_indices) if class_indices is not None else None
        for class_name in self.class_names:
            class_idx = self.class_to_idx[class_name]
            if allowed is not None and class_idx not in allowed:
                continue
            split_dir = os.path.join(self.modelnet_dir, class_name, split)
            if not os.path.isdir(split_dir):
                continue
            for fname in sorted(os.listdir(split_dir)):
                if fname.endswith(".off"):
                    path = os.path.join(split_dir, fname)
                    self.files.append((path, class_idx))
        if not self.files:
            raise RuntimeError(f"No ModelNet40 data found for split '{split}'.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path, label = self.files[idx]
        verts, faces = read_off(path)
        points = sample_points_from_mesh(verts, faces, self.num_points)
        points = normalize_point_cloud(points)
        if self.augment:
            points = random_rotate_z(points)
            points = random_scale(points)
            points = random_shift(points)
            points = jitter_points(points)
            points = random_point_dropout(points)
        return torch.from_numpy(points), label


class PointNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        features = self.mlp(x)
        pooled = torch.max(features, dim=2).values
        return pooled


class PointNetClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.encoder = PointNetEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.classifier(features)


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
            points = points.to(next(model.parameters()).device)
            labels = labels.to(points.device)
            logits = model(points)
            preds = logits.argmax(dim=1)
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
    device: torch.device,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    model.train()
    for epoch in range(EPOCHS):
        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for points, labels in progress:
            points = points.to(device)
            labels = labels.to(device)
            logits = model(points)
            ce = F.cross_entropy(logits, labels)
            if old_model is None:
                loss = ce
            else:
                with torch.no_grad():
                    old_logits = old_model(points)
                    p_old = F.softmax(old_logits, dim=1)
                log_p_new = F.log_softmax(logits, dim=1)
                kl = F.kl_div(log_p_new, p_old, reduction="batchmean")
                loss = ce + LWF_LAMBDA * kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def main():
    set_deterministic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tasks = [
        list(range(0, 10)),
        list(range(10, 20)),
        list(range(20, 30)),
        list(range(30, 40)),
    ]

    model = PointNetClassifier(NUM_CLASSES).to(device)
    max_class_acc: Dict[int, float] = {}

    for task_id, task_classes in enumerate(tasks, start=1):
        train_dataset = ModelNet40(
            DATA_DIR, split="train", num_points=NUM_POINTS, class_indices=task_classes
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

        train_task(model, train_loader, old_model, device)

        seen_classes = sum(tasks[:task_id], [])
        test_dataset = ModelNet40(
            DATA_DIR, split="test", num_points=NUM_POINTS, class_indices=seen_classes
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
