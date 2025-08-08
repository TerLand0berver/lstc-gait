import argparse
from dataclasses import dataclass
from pathlib import Path
import random
import os

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from lstc import LSTCBackbone


@dataclass
class ToyConfig:
    num_classes: int = 10
    seq_len: int = 16
    height: int = 64
    width: int = 44
    channels: int = 1


class ToyGaitDataset(Dataset):
    def __init__(self, num_samples: int, cfg: ToyConfig, seed: int = 42):
        super().__init__()
        self.num_samples = num_samples
        self.cfg = cfg
        random.seed(seed)
        self.labels = [random.randint(0, cfg.num_classes - 1) for _ in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        cfg = self.cfg
        label = self.labels[idx]
        # Synthesize a simple sequence with class-dependent mean pattern
        torch.manual_seed(idx + label * 1000)
        x = torch.randn(cfg.channels, cfg.seq_len, cfg.height, cfg.width) * 0.5
        x += (label / max(1, cfg.num_classes - 1)) * 0.5
        return x, label


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / max(1, targets.numel())


def save_checkpoint(state: dict, out_dir: Path, is_best: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ("best.pt" if is_best else "last.pt")
    torch.save(state, path)


def train_one_epoch(model, criterion, optimizer, loader, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        emb = out["embedding"]
        logits = classifier(emb)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        running_acc += accuracy(logits.detach(), y) * x.size(0)

    n = len(loader.dataset)
    return running_loss / n, running_acc / n


def evaluate(model, criterion, loader, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            emb = out["embedding"]
            logits = classifier(emb)
            loss = criterion(logits, y)
            running_loss += loss.item() * x.size(0)
            running_acc += accuracy(logits, y) * x.size(0)
    n = len(loader.dataset)
    return running_loss / n, running_acc / n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=44)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--num-stripes", type=int, default=8)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="runs/lstc_toy")
    args = parser.parse_args()

    device = torch.device(args.device)

    cfg = ToyConfig(
        num_classes=args.num_classes,
        seq_len=args.seq_len,
        height=args.height,
        width=args.width,
        channels=1,
    )

    train_set = ToyGaitDataset(num_samples=512, cfg=cfg, seed=123)
    val_set = ToyGaitDataset(num_samples=128, cfg=cfg, seed=321)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = LSTCBackbone(
        in_channels=cfg.channels,
        base_channels=args.base_channels,
        num_stripes=args.num_stripes,
        embedding_dim=args.embedding_dim,
    ).to(device)

    global classifier
    classifier = nn.Linear(args.embedding_dim, args.num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    best_acc = 0.0
    out_dir = Path(args.out_dir)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, criterion, optimizer, train_loader, device)
        va_loss, va_acc = evaluate(model, criterion, val_loader, device)
        print(f"Epoch {epoch}: train loss={tr_loss:.4f} acc={tr_acc:.3f} | val loss={va_loss:.4f} acc={va_acc:.3f}")

        save_checkpoint({
            "model": model.state_dict(),
            "classifier": classifier.state_dict(),
            "epoch": epoch,
            "val_acc": va_acc,
        }, out_dir, is_best=va_acc >= best_acc)
        best_acc = max(best_acc, va_acc)

    print(f"Best val acc: {best_acc:.3f}")
