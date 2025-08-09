import argparse
import os
from pathlib import Path
from typing import Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

from lstc import LSTCBackbone
from lstc.losses import BatchHardTripletLoss
from lstc.samplers import PKSampler
from lstc.utils import set_seed, is_main_process, CSVLogger, save_checkpoint, try_load_checkpoint, create_tb_writer, ModelEma
from examples.dataset_silhouette import scan_silhouette_root, GaitSilhouetteDataset
from omegaconf import OmegaConf


def build_dataloader_pk(data_root: Path, seq_len: int, height: int, width: int, batch_p: int, batch_k: int, num_workers: int) -> Tuple[DataLoader, int]:
    records, label_map = scan_silhouette_root(data_root, min_frames=2)
    dataset = GaitSilhouetteDataset(records, seq_len=seq_len, height=height, width=width, sampling="uniform")
    labels = [r.label_id for r in records]
    sampler = PKSampler(labels=labels, batch_p=batch_p, batch_k=batch_k)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, pin_memory=True)
    num_classes = len(label_map)
    return loader, num_classes


def train_epoch(model, classifier, ce_loss, tri_loss, loader, optimizer, device, amp: bool):
    model.train(); classifier.train()
    use_cuda_amp = amp and (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda') if use_cuda_amp else torch.amp.GradScaler(enabled=False)
    running = {"ce": 0.0, "tri": 0.0}
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=use_cuda_amp):
            out = model(x)
            emb = out["embedding"]
            logits = classifier(emb)
            loss_ce = ce_loss(logits, y)
            loss_tri = tri_loss(emb, y)
            loss = loss_ce + loss_tri
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running["ce"] += loss_ce.item() * x.size(0)
        running["tri"] += loss_tri.item() * x.size(0)
    n = len(loader.dataset)
    return {k: v / n for k, v in running.items()}


def evaluate(model, classifier, ce_loss, loader, device):
    model.eval(); classifier.eval()
    running_ce = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(x)
            emb = out["embedding"]
            logits = classifier(emb)
            running_ce += ce_loss(logits, y).item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return {"ce": running_ce / max(1, total), "acc": correct / max(1, total)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="Path to YAML config")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=44)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--num-stripes", type=int, default=8)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--pooling-topk", type=int, default=2)
    parser.add_argument("--pooling-soft", action="store_true")
    parser.add_argument("--pooling-temperature", type=float, default=1.0)
    parser.add_argument("--use-temporal", action="store_true", default=True)
    parser.add_argument("--use-spatial", action="store_true", default=True)
    parser.add_argument("--use-joint", action="store_true", default=True)
    parser.add_argument("--batch-p", type=int, default=8)
    parser.add_argument("--batch-k", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--out-dir", type=str, default="runs/lstc_metric")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--csv-log", action="store_true")
    parser.add_argument("--log-dir", type=str, default="runs/logs_metric")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--ddp", action="store_true", help="enable DistributedDataParallel; launch with torchrun")
    args = parser.parse_args()

    if args.config:
        cfg = OmegaConf.load(args.config)
        cli = OmegaConf.create(vars(args))
        cfg = OmegaConf.merge(cfg, cli)
        args = argparse.Namespace(**OmegaConf.to_container(cfg, resolve=True))

    set_seed(args.seed)

    # DDP setup
    if args.ddp:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
    else:
        device = torch.device(args.device)
    data_root = Path(args.data_root)

    base_loader, num_classes = build_dataloader_pk(
        data_root=data_root,
        seq_len=args.seq_len,
        height=args.height,
        width=args.width,
        batch_p=args.batch_p,
        batch_k=args.batch_k,
        num_workers=args.num_workers,
    )
    # Wrap PK sampler batches for DDP (optional): if sampler yields lists, set batch_sampler directly
    loader = base_loader

    model = LSTCBackbone(
        in_channels=1,
        base_channels=args.base_channels,
        num_stripes=args.num_stripes,
        embedding_dim=args.embedding_dim,
        pooling_topk=args.pooling_topk,
        pooling_soft=args.pooling_soft,
        pooling_temperature=args.pooling_temperature,
        use_temporal=args.use_temporal,
        use_spatial=args.use_spatial,
        use_joint=args.use_joint,
    ).to(device)
    classifier = nn.Linear(args.embedding_dim, num_classes).to(device)

    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
        classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[device.index] if device.type == "cuda" else None)

    ce_loss = nn.CrossEntropyLoss()
    tri_loss = BatchHardTripletLoss(margin=0.3, metric="euclidean")

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Resume
    start_epoch = 1
    best = {"acc": 0.0}
    ckpt = try_load_checkpoint(args.resume)
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        classifier.load_state_dict(ckpt["classifier"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])  # type: ignore[arg-type]
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])  # type: ignore[arg-type]
        start_epoch = ckpt.get("epoch", 0) + 1
        if "best" in ckpt:
            best = ckpt["best"]

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    tb_writer = create_tb_writer(log_dir / "tb", enabled=(args.tensorboard and (not args.ddp or dist.get_rank() == 0)))
    csv_logger = CSVLogger(log_dir / "train.csv", ["epoch", "train_ce", "train_tri", "val_ce", "val_acc", "lr"]) if (args.csv_log and (not args.ddp or dist.get_rank() == 0)) else None

    ema = ModelEma(model if not isinstance(model, nn.parallel.DistributedDataParallel) else model.module, decay=args.ema_decay) if args.ema else None

    for epoch in range(start_epoch, args.epochs + 1):
        train_stats = train_epoch(model, classifier, ce_loss, tri_loss, loader, optimizer, device, amp=args.amp)
        # Grad clip
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(classifier.parameters()), max_norm=args.grad_clip)
        # EMA update
        if ema is not None:
            ema.update(model if not isinstance(model, nn.parallel.DistributedDataParallel) else model.module)
        eval_stats = evaluate(model, classifier, ce_loss, loader, device)
        eval_ema_acc = None
        if ema is not None and ((not args.ddp) or dist.get_rank() == 0):
            eval_ema = evaluate(ema.module, classifier, ce_loss, loader, device)
            eval_ema_acc = eval_ema["acc"]
        scheduler.step()
        if (not args.ddp) or dist.get_rank() == 0:
            msg = f"Epoch {epoch:03d} | train ce={train_stats['ce']:.4f} tri={train_stats['tri']:.4f} | val ce={eval_stats['ce']:.4f} acc={eval_stats['acc']:.3f}"
            if eval_ema_acc is not None:
                msg += f" | ema acc={eval_ema_acc:.3f}"
            print(msg)

        if tb_writer:
            tb_writer.add_scalar("train/ce", train_stats['ce'], epoch)
            tb_writer.add_scalar("train/tri", train_stats['tri'], epoch)
            tb_writer.add_scalar("val/ce", eval_stats['ce'], epoch)
            tb_writer.add_scalar("val/acc", eval_stats['acc'], epoch)
            tb_writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
        if csv_logger:
            csv_logger.log({
                "epoch": epoch,
                "train_ce": train_stats['ce'],
                "train_tri": train_stats['tri'],
                "val_ce": eval_stats['ce'],
                "val_acc": eval_stats['acc'],
                "lr": optimizer.param_groups[0]['lr'],
            })

        # Save last
        state = {
            "model": model.state_dict(),
            "classifier": classifier.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best": best,
            "args": vars(args),
            "num_classes": num_classes,
        }
        if (not args.ddp) or dist.get_rank() == 0:
            save_checkpoint(state, out_dir / "last.pt")

            if eval_stats["acc"] >= best["acc"]:
                best = eval_stats
                save_checkpoint(state, out_dir / "best.pt")
            if eval_ema_acc is not None:
                state_ema = dict(state)
                state_ema["model"] = (ema.module).state_dict()
                state_ema["best"] = {"acc": eval_ema_acc}
                save_checkpoint(state_ema, out_dir / "last_ema.pt")
                # Track best ema acc in loop-local var (not persisted across restarts)
                if "_best_ema_acc" not in locals():
                    _best_ema_acc = 0.0
                if eval_ema_acc >= _best_ema_acc:
                    _best_ema_acc = eval_ema_acc
                    save_checkpoint(state_ema, out_dir / "best_ema.pt")

    if tb_writer:
        tb_writer.close()
    if csv_logger:
        csv_logger.close()
    if (not args.ddp) or dist.get_rank() == 0:
        tail = f", best ema acc: {_best_ema_acc:.3f}" if ('_best_ema_acc' in locals()) else ""
        print(f"Best acc: {best['acc']:.3f}{tail}")
