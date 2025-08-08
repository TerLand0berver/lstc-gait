import argparse
import os
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from lstc import LSTCBackbone
from lstc.utils import (
    set_seed,
    is_main_process,
    CSVLogger,
    save_checkpoint,
    try_load_checkpoint,
    create_tb_writer,
    ModelEma,
)
from examples.dataset_silhouette import scan_silhouette_root, GaitSilhouetteDataset
from omegaconf import OmegaConf


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / max(1, targets.numel())


def build_dataloaders(
    data_root: Path,
    seq_len: int,
    height: int,
    width: int,
    batch_size: int,
    num_workers: int,
    split_ratio: float = 0.9,
    distributed: bool = False,
) -> Tuple[DataLoader, DataLoader, int, Optional[DistributedSampler], Optional[DistributedSampler]]:
    records, label_map = scan_silhouette_root(data_root, min_frames=2)
    if len(records) < 2:
        raise RuntimeError(f"Not enough sequences found under {data_root} (found {len(records)})")

    dataset = GaitSilhouetteDataset(records, seq_len=seq_len, height=height, width=width, sampling="uniform")
    num_classes = len(label_map)

    n_train = int(len(dataset) * split_ratio)
    n_val = max(1, len(dataset) - n_train)
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_sampler = DistributedSampler(train_set, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, num_classes, train_sampler, val_sampler


def train_one_epoch(model, classifier, criterion, optimizer, loader, device, amp: bool = False):
    model.train()
    classifier.train()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    loss_sum = torch.tensor(0.0, device=device)
    correct_sum = torch.tensor(0.0, device=device)
    total_num = torch.tensor(0.0, device=device)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            out = model(x)
            emb = out["embedding"]
            logits = classifier(emb)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.detach() * x.size(0)
        correct_sum += (logits.detach().argmax(dim=1) == y).sum()
        total_num += x.size(0)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_num, op=dist.ReduceOp.SUM)

    loss_avg = (loss_sum / torch.clamp(total_num, min=1)).item()
    acc_avg = (correct_sum / torch.clamp(total_num, min=1)).item()
    return loss_avg, acc_avg


def evaluate(model, classifier, criterion, loader, device):
    model.eval()
    classifier.eval()
    loss_sum = torch.tensor(0.0, device=device)
    correct_sum = torch.tensor(0.0, device=device)
    total_num = torch.tensor(0.0, device=device)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(x)
            emb = out["embedding"]
            logits = classifier(emb)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss_sum += loss * x.size(0)
            correct_sum += (logits.argmax(dim=1) == y).sum()
            total_num += x.size(0)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_num, op=dist.ReduceOp.SUM)

    loss_avg = (loss_sum / torch.clamp(total_num, min=1)).item()
    acc_avg = (correct_sum / torch.clamp(total_num, min=1)).item()
    return loss_avg, acc_avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="Path to YAML config")
    parser.add_argument("--data-root", type=str, required=False, help="Path to dataset root directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=44)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--num-stripes", type=int, default=8)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out-dir", type=str, default="runs/lstc_real")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--csv-log", action="store_true")
    parser.add_argument("--log-dir", type=str, default="runs/logs")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--early-stop", type=int, default=0, help="patience in epochs (0 disables)")
    parser.add_argument("--amp", action="store_true", help="use mixed precision")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--ddp", action="store_true", help="enable DistributedDataParallel; launch with torchrun")
    args = parser.parse_args()

    if args.config:
        cfg = OmegaConf.load(args.config)
        cli = OmegaConf.create(vars(args))
        cfg = OmegaConf.merge(cfg, cli)  # CLI overrides YAML
        args = argparse.Namespace(**OmegaConf.to_container(cfg, resolve=True))

    set_seed(args.seed)

    if args.ddp:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
    else:
        device = torch.device(args.device)

    if not args.data_root:
        raise SystemExit("--data-root must be provided (via CLI or config)")
    data_root = Path(args.data_root)

    train_loader, val_loader, num_classes, train_sampler, val_sampler = build_dataloaders(
        data_root=data_root,
        seq_len=args.seq_len,
        height=args.height,
        width=args.width,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed=args.ddp,
    )

    model = LSTCBackbone(
        in_channels=1,
        base_channels=args.base_channels,
        num_stripes=args.num_stripes,
        embedding_dim=args.embedding_dim,
    ).to(device)
    classifier = nn.Linear(args.embedding_dim, num_classes).to(device)

    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
        classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[device.index] if device.type == "cuda" else None)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 1
    best_acc = 0.0
    ckpt = try_load_checkpoint(args.resume)
    if ckpt is not None:
        (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model).load_state_dict(ckpt["model"])
        (classifier.module if isinstance(classifier, torch.nn.parallel.DistributedDataParallel) else classifier).load_state_dict(ckpt["classifier"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])  # type: ignore[arg-type]
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])  # type: ignore[arg-type]
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("val_acc", ckpt.get("best", {}).get("acc", 0.0))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    tb_writer = create_tb_writer(log_dir / "tb", enabled=(args.tensorboard and is_main_process()))
    csv_logger = CSVLogger(log_dir / "train.csv", ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]) if (args.csv_log and is_main_process()) else None

    patience = args.early_stop
    bad_epochs = 0

    ema = ModelEma(model if not isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.module, decay=args.ema_decay) if args.ema else None

    for epoch in range(start_epoch, args.epochs + 1):
        if args.ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        tr_loss, tr_acc = train_one_epoch(model, classifier, criterion, optimizer, train_loader, device, amp=args.amp)
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(classifier.parameters()), max_norm=args.grad_clip)
        if ema is not None:
            ema.update(model if not isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.module)
        va_loss, va_acc = evaluate(model, classifier, criterion, val_loader, device)
        scheduler.step()

        if is_main_process():
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

            state = {
                "model": (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model).state_dict(),
                "classifier": (classifier.module if isinstance(classifier, torch.nn.parallel.DistributedDataParallel) else classifier).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "val_acc": va_acc,
                "num_classes": num_classes,
                "args": vars(args),
            }
            save_checkpoint(state, out_dir / "last.pt")

            improved = va_acc >= best_acc
            if improved:
                best_acc = va_acc
                save_checkpoint(state, out_dir / "best.pt")
                bad_epochs = 0
            else:
                bad_epochs += 1

        stop_flag = torch.tensor(1 if (patience > 0 and bad_epochs >= patience and is_main_process()) else 0, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(stop_flag, src=0)
        if int(stop_flag.item()) == 1:
            break

    if tb_writer:
        tb_writer.close()
    if csv_logger:
        csv_logger.close()
    if is_main_process():
        print(f"Best val acc: {best_acc:.3f}")
