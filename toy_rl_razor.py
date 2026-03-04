#!/usr/bin/env python3
"""Continual-learning toy from *RL's Razor* showing SFT vs. RL across three tasks."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import FashionMNIST, MNIST


def parse_args():
    parser = argparse.ArgumentParser(
        description="Toy continual learning experiment based on RL's Razor."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for base and per-task training.",
    )
    parser.add_argument(
        "--task-epochs",
        type=int,
        default=2,
        help="Number of epochs to train on each continual task.",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=4096,
        help="Upper bound on how many MNIST/Fashion samples to use.",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=256,
        help="Hidden width of the shared MLP.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on (`cpu`, `cuda`, or `auto`).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["sft", "rl"],
        default=["sft", "rl"],
        help="Which continual-training methods to run (`sft` for supervised fine-tuning, `rl` for REINFORCE-style updates).",
    )
    return parser.parse_args()


def choose_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


class ToyMLP(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TaskSpec:
    """Definition of one new task (meta labels over digits)."""

    name: str
    label_groups: Dict[int, Sequence[int]]

    def __post_init__(self):
        if sum(len(v) for v in self.label_groups.values()) != 10:
            raise ValueError("label_groups must cover digits 0-9 exactly once.")
        self.meta_lookup = torch.zeros(10, dtype=torch.long)
        for meta, digits in self.label_groups.items():
            for d in digits:
                self.meta_lookup[d] = meta

    def sample_labels(self, digits: torch.Tensor) -> torch.Tensor:
        digits = digits.to("cpu")
        sampled = []
        for d in digits.tolist():
            meta = self.meta_lookup[d].item()
            sampled.append(random.choice(list(self.label_groups[meta])))
        return torch.tensor(sampled, dtype=torch.long)

    def reward(self, actions: torch.Tensor, digits: torch.Tensor) -> torch.Tensor:
        lookup = self.meta_lookup.to(actions.device)
        return (lookup[actions] == lookup[digits]).float()

    def meta_accuracy(self, logits: torch.Tensor, digits: torch.Tensor) -> float:
        lookup = self.meta_lookup.to(logits.device)
        preds = logits.argmax(dim=1)
        correct = (lookup[preds] == lookup[digits]).sum().item()
        return correct / digits.size(0)


def build_datasets(subset: int):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    mnist_train = MNIST(root="data", train=True, download=True, transform=transform)
    fashion_train = FashionMNIST(
        root="data", train=True, download=True, transform=transform
    )
    mnist_test = MNIST(
        root="data", train=False, download=True, transform=transform
    )
    fashion_test = FashionMNIST(
        root="data", train=False, download=True, transform=transform
    )

    train_subset = list(range(min(subset, len(mnist_train))))
    eval_subset = list(range(min(subset // 2, len(mnist_test))))

    return (
        Subset(mnist_train, train_subset),
        Subset(fashion_train, train_subset),
        Subset(mnist_test, eval_subset),
        Subset(fashion_test, eval_subset),
    )


def evaluate_fashion(model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, labels in dataloader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == labels.to(device)).sum().item()
            total += labels.size(0)
    return correct / total


def compute_kl(fine_model: nn.Module, base_model: nn.Module, dataloader: DataLoader, device: torch.device):
    fine_model.eval()
    base_model.eval()
    total_kl = 0.0
    total = 0
    eps = 1e-8
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            logp = F.log_softmax(fine_model(x), dim=1)
            q = F.softmax(base_model(x), dim=1).clamp(min=eps)
            p = logp.exp().clamp(min=eps)
            kl = (p * (logp - torch.log(q))).sum(dim=1).sum().item()
            total_kl += kl
            total += x.size(0)
    return total_kl / total


def train_sft_on_task(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    task: TaskSpec,
):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    model.train()
    for _ in range(epochs):
        for x, digits in loader:
            x = x.to(device)
            digits = digits.to(device)
            targets = task.sample_labels(digits).to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def train_rl_on_task(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    task: TaskSpec,
):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    model.train()
    baseline = 0.5
    for _ in range(epochs):
        for x, digits in loader:
            x = x.to(device)
            digits = digits.to(device)
            logits = model(x)
            dist = Categorical(logits=logits)
            actions = dist.sample()
            rewards = task.reward(actions, digits)
            advantages = rewards - baseline
            loss = - (advantages.detach() * dist.log_prob(actions)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            baseline = 0.9 * baseline + 0.1 * rewards.mean().item()


def train_base(model: nn.Module, dataloader: DataLoader, device: torch.device, epochs: int):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    for _ in range(epochs):
        for x, labels in dataloader:
            x = x.to(device)
            labels = labels.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def run_continual(
    base_model: nn.Module,
    fashion_eval_loader: DataLoader,
    digits_eval_loader: DataLoader,
    digit_dataset: Dataset,
    device: torch.device,
    args: argparse.Namespace,
    tasks: Iterable[TaskSpec],
):
    methods: Dict[str, nn.Module] = {}
    for method in args.methods:
        label = "SFT" if method == "sft" else "RL"
        methods[label] = ToyMLP(args.hidden).to(device)
        methods[label].load_state_dict(base_model.state_dict())

    history: Dict[str, List[Dict[str, float]]] = {name: [] for name in methods}

    for task in tasks:
        for name, model in methods.items():
            if name == "SFT":
                train_sft_on_task(
                    model,
                    digit_dataset,
                    device,
                    args.task_epochs,
                    args.batch_size,
                    task,
                )
            else:
                train_rl_on_task(
                    model,
                    digit_dataset,
                    device,
                    args.task_epochs,
                    args.batch_size,
                    task,
                )

        for name, model in methods.items():
            new_acc = evaluate_task(model, digits_eval_loader, task, device)
            prior_acc = evaluate_fashion(model, fashion_eval_loader, device)
            kl_shift = compute_kl(model, base_model, digits_eval_loader, device)
            history[name].append(
                {
                    "task": task.name,  # type: ignore[arg-type]
                    "new_acc": new_acc,
                    "prior_acc": prior_acc,
                    "kl_shift": kl_shift,
                }
            )

    return history


def evaluate_task(model: nn.Module, dataloader: DataLoader, task: TaskSpec, device: torch.device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, digits in dataloader:
            x = x.to(device)
            digits = digits.to(device)
            logits = model(x)
            correct += task.meta_accuracy(logits, digits) * digits.size(0)
            total += digits.size(0)
    return correct / total


def format_history(history: Dict[str, List[Dict[str, float]]]):
    lines = []
    header = f"{'Task':>10} | {'Method':>6} | {'New Acc':>7} | {'Prior Acc':>9} | {'KL Shift':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for name, records in history.items():
        for record in records:
            lines.append(
                f"{record['task']:>10} | {name:>6} | {record['new_acc']*100:7.2f}% | "
                f"{record['prior_acc']*100:9.2f}% | {record['kl_shift']:8.4f}"
            )
    return "\n".join(lines)


def main():
    args = parse_args()
    device = choose_device(args.device)
    torch.manual_seed(0)
    random.seed(0)

    (
        digits_train,
        fashion_train,
        digits_eval,
        fashion_eval,
    ) = build_datasets(args.subset)

    fashion_loader = DataLoader(fashion_train, batch_size=args.batch_size, shuffle=True)
    fashion_eval_loader = DataLoader(fashion_eval, batch_size=args.batch_size)
    digits_eval_loader = DataLoader(digits_eval, batch_size=args.batch_size)

    base_model = ToyMLP(args.hidden).to(device)
    train_base(base_model, fashion_loader, device, epochs=3)

    base_prior = evaluate_fashion(base_model, fashion_eval_loader, device)

    tasks = [
        TaskSpec("Parity", {0: [0, 2, 4, 6, 8], 1: [1, 3, 5, 7, 9]}),
        TaskSpec("Mod3", {0: [0, 3, 6, 9], 1: [1, 4, 7], 2: [2, 5, 8]}),
        TaskSpec("HighLow", {0: [0, 1, 2, 3, 4], 1: [5, 6, 7, 8, 9]}),
    ]

    history = run_continual(
        base_model,
        fashion_eval_loader,
        digits_eval_loader,
        digits_train,
        device,
        args,
        tasks,
    )

    print(f"Base prior (Fashion) accuracy: {base_prior*100:.1f}%")
    print(format_history(history))


if __name__ == "__main__":
    main()
