import argparse
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from lstc import LSTCBackbone
from lstc.utils import set_seed, is_main_process, CSVLogger, save_checkpoint, try_load_checkpoint, create_tb_writer
from examples.dataset_multiview import MultiViewGaitDataset
from lstc.samplers import MultiViewPKSampler
from omegaconf import OmegaConf


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / max(1, targets.numel())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to multiview YAML config")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    args = argparse.Namespace(**OmegaConf.to_container(cfg, resolve=True))

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not getattr(args, "data_roots", None):
        raise SystemExit("configs/multiview_real.yaml requires 'data_roots' list")

    dataset = MultiViewGaitDataset([Path(p) for p in args.data_roots], seq_len=args.seq_len, height=args.height, width=args.width)
    n_train = int(len(dataset) * args.split_ratio)
    n_val = max(1, len(dataset) - n_train)
    train_set, val_set = random_split(dataset, [n_train, n_val])

    # Multi-view PK sampling for better cross-view mixing (auto-fit batch_p/batch_k)
    train_labels = []
    train_views = []
    for i in range(len(train_set)):
        _, y, v = train_set[i]
        train_labels.append(int(y)); train_views.append(int(v))
    num_ids = len(set(train_labels))
    batch_p = max(2, min(8, num_ids))
    batch_k = max(1, args.batch_size // batch_p)
    sampler = MultiViewPKSampler(
        labels=train_labels,
        view_ids=train_views,
        batch_p=batch_p,
        batch_k=batch_k,
        views_per_id=2,
        balance_across_views=True,
    )
    train_loader = DataLoader(train_set, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    model = LSTCBackbone(in_channels=1, base_channels=args.base_channels, num_stripes=args.num_stripes, embedding_dim=args.embedding_dim).to(device)
    num_classes = len(dataset.global_label_map)
    classifier = nn.Linear(args.embedding_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    tb_writer = create_tb_writer(log_dir / "tb", enabled=args.tensorboard)
    csv_logger = CSVLogger(log_dir / "train.csv", ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]) if args.csv_log else None

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        # train
        model.train(); classifier.train()
        loss_sum = 0.0; correct = 0; total = 0
        for x, y, view_id in train_loader:
            x = x.to(device); y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            emb = out["embedding"]
            logits = classifier(emb)
            loss = criterion(logits, y)
            loss.backward(); optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (logits.detach().argmax(dim=1) == y).sum().item()
            total += x.size(0)
        tr_loss, tr_acc = loss_sum / max(1, total), correct / max(1, total)

        # val
        model.eval(); classifier.eval(); loss_sum = 0.0; correct = 0; total = 0
        with torch.no_grad():
            for x, y, view_id in val_loader:
                x = x.to(device); y = y.to(device)
                out = model(x)
                emb = out["embedding"]
                logits = classifier(emb)
                loss = criterion(logits, y)
                loss_sum += loss.item() * x.size(0)
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += x.size(0)
        va_loss, va_acc = loss_sum / max(1, total), correct / max(1, total)
        scheduler.step()

        print(f"Epoch {epoch}: train loss={tr_loss:.4f} acc={tr_acc:.3f} | val loss={va_loss:.4f} acc={va_acc:.3f}")
        if tb_writer:
            tb_writer.add_scalar("train/loss", tr_loss, epoch)
            tb_writer.add_scalar("train/acc", tr_acc, epoch)
            tb_writer.add_scalar("val/loss", va_loss, epoch)
            tb_writer.add_scalar("val/acc", va_acc, epoch)
            tb_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        if csv_logger:
            csv_logger.log({
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": va_loss,
                "val_acc": va_acc,
                "lr": optimizer.param_groups[0]["lr"],
            })

        if va_acc >= best_acc:
            best_acc = va_acc
            ckpt = {"model": model.state_dict(), "classifier": classifier.state_dict(), "epoch": epoch, "val_acc": va_acc, "args": vars(args)}
            torch.save(ckpt, out_dir / "best.pt")

    if tb_writer: tb_writer.close()
    if csv_logger: csv_logger.close()
    print(f"Best val acc: {best_acc:.3f}")
