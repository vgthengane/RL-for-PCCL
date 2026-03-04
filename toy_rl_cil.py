#!/usr/bin/env python3

import copy
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.distributions import Categorical
from torchvision.datasets import MNIST
from torchvision import transforms


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256
EPOCHS = 2
LR = 1e-4
KL_COEF = 0.05
EWC_LAMBDA = 10.0
EWC_BATCHES = 10


# ------------------------------------------------
# Model
# ------------------------------------------------

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,10)
        )

    def forward(self,x):
        return self.net(x)


# ------------------------------------------------
# Dataset utilities
# ------------------------------------------------

def filter_classes(dataset, classes):

    idx = [i for i,(x,y) in enumerate(dataset) if y in classes]

    return Subset(dataset, idx)


# ------------------------------------------------
# Evaluation
# ------------------------------------------------

def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for x,y in loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)

            pred = logits.argmax(1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total


# ------------------------------------------------
# KL preservation
# ------------------------------------------------

def kl_loss(old_logits, new_logits):

    p = F.softmax(old_logits, dim=1)
    logp = F.log_softmax(old_logits, dim=1)

    logq = F.log_softmax(new_logits, dim=1)

    return (p * (logp - logq)).sum(dim=1).mean()


# ------------------------------------------------
# Elastic Weight Consolidation (EWC)
# ------------------------------------------------

def snapshot_params(model):
    return {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}


def compute_fisher(model, loader):
    model.eval()
    fisher = {
        name: torch.zeros_like(p, device=DEVICE)
        for name, p in model.named_parameters()
        if p.requires_grad
    }
    batch_count = 0
    for x, y in loader:
        if batch_count >= EWC_BATCHES:
            break
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        model.zero_grad()
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                fisher[name] += p.grad.detach() ** 2
        batch_count += 1
    if batch_count == 0:
        return fisher
    for name in fisher:
        fisher[name] /= batch_count
    return fisher


def ewc_penalty(model, ewc_list):
    if not ewc_list:
        return torch.tensor(0.0, device=DEVICE)
    loss = torch.tensor(0.0, device=DEVICE)
    for ewc in ewc_list:
        fisher = ewc["fisher"]
        params = ewc["params"]
        for name, p in model.named_parameters():
            if name in fisher:
                loss += (fisher[name] * (p - params[name]) ** 2).sum()
    return loss


# ------------------------------------------------
# SFT training
# ------------------------------------------------

def train_sft(model, loader):

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for _ in range(EPOCHS):

        for x,y in loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)

            loss = F.cross_entropy(logits,y)

            opt.zero_grad()
            loss.backward()
            opt.step()


# ------------------------------------------------
# SFT + KL
# ------------------------------------------------

def train_sft_kl(model, old_model, loader):

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    old_model.eval()

    for _ in range(EPOCHS):

        for x,y in loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            with torch.no_grad():
                old_logits = old_model(x)

            logits = model(x)

            ce = F.cross_entropy(logits,y)

            kl = kl_loss(old_logits, logits)

            loss = ce + KL_COEF * kl

            opt.zero_grad()
            loss.backward()
            opt.step()


# ------------------------------------------------
# SFT + EWC
# ------------------------------------------------

def train_sft_ewc(model, loader, ewc_list):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(EPOCHS):
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            logits = model(x)
            ce = F.cross_entropy(logits, y)
            penalty = ewc_penalty(model, ewc_list)
            loss = ce + EWC_LAMBDA * penalty
            opt.zero_grad()
            loss.backward()
            opt.step()


# ------------------------------------------------
# RL training
# ------------------------------------------------

def train_rl(model, loader):

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for _ in range(EPOCHS):

        for x,y in loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)

            dist = Categorical(logits=logits)

            actions = dist.sample()

            reward = (actions == y).float()

            advantage = reward - reward.mean()

            loss = -(advantage.detach()*dist.log_prob(actions)).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()


# ------------------------------------------------
# RL + KL
# ------------------------------------------------

def train_rl_kl(model, old_model, loader):

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    old_model.eval()

    for _ in range(EPOCHS):

        for x,y in loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)

            dist = Categorical(logits=logits)

            actions = dist.sample()

            reward = (actions == y).float()

            advantage = reward - reward.mean()

            policy_loss = -(advantage.detach()*dist.log_prob(actions)).mean()

            with torch.no_grad():
                old_logits = old_model(x)

            kl = kl_loss(old_logits, logits)

            loss = policy_loss + KL_COEF * kl

            opt.zero_grad()
            loss.backward()
            opt.step()


# ------------------------------------------------
# RL + EWC
# ------------------------------------------------

def train_rl_ewc(model, loader, ewc_list):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(EPOCHS):
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            logits = model(x)
            dist = Categorical(logits=logits)
            actions = dist.sample()
            reward = (actions == y).float()
            advantage = reward - reward.mean()
            policy_loss = -(advantage.detach() * dist.log_prob(actions)).mean()
            penalty = ewc_penalty(model, ewc_list)
            loss = policy_loss + EWC_LAMBDA * penalty
            opt.zero_grad()
            loss.backward()
            opt.step()


# ------------------------------------------------
# Main experiment
# ------------------------------------------------

def main():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train = MNIST("./datasets",train=True,download=True,transform=transform)
    test  = MNIST("./datasets",train=False,download=True,transform=transform)

    tasks = [
        [0,1,2],
        [3,4,5],
        [6,7],
        [8,9]
    ]

    models = {
        "SFT": MLP().to(DEVICE),
        "SFT+KL": MLP().to(DEVICE),
        "SFT+EWC": MLP().to(DEVICE),
        "RL": MLP().to(DEVICE),
        "RL+KL": MLP().to(DEVICE),
        "RL+EWC": MLP().to(DEVICE)
    }

    ewc_states = {name: [] for name in models}

    for t,classes in enumerate(tasks):

        print(f"\n=== Task {t+1} classes {classes} ===")

        train_subset = filter_classes(train, classes)
        loader = DataLoader(train_subset,batch_size=BATCH_SIZE,shuffle=True)

        for name,model in models.items():

            old_model = copy.deepcopy(model)

            if name=="SFT":
                train_sft(model,loader)

            elif name=="SFT+KL":
                train_sft_kl(model,old_model,loader)

            elif name=="SFT+EWC":
                train_sft_ewc(model,loader,ewc_states[name])

            elif name=="RL":
                train_rl(model,loader)

            elif name=="RL+KL":
                train_rl_kl(model,old_model,loader)

            elif name=="RL+EWC":
                train_rl_ewc(model,loader,ewc_states[name])

            fisher = compute_fisher(model, loader)
            ewc_states[name].append(
                {"params": snapshot_params(model), "fisher": fisher}
            )

        test_subset = filter_classes(test, sum(tasks[:t+1],[]))
        test_loader = DataLoader(test_subset,batch_size=BATCH_SIZE)

        for name,model in models.items():

            acc = evaluate(model,test_loader)

            print(f"{name:8s} accuracy: {acc*100:.2f}%")


if __name__ == "__main__":
    main()