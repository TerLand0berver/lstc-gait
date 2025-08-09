import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, random_split

from lstc import LSTCBackbone
from lstc.losses import BatchHardTripletLoss
from lstc.samplers import MultiViewPKSampler
from lstc.utils import set_seed, CSVLogger, create_tb_writer, save_checkpoint, try_load_checkpoint
from examples.dataset_multiview import MultiViewGaitDataset
from omegaconf import OmegaConf


def train_epoch(model, classifier, ce_loss, tri_loss, loader, optimizer, device, amp: bool):
    model.train(); classifier.train()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    running = {"ce": 0.0, "tri": 0.0}
    for x, y, view_id in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            out = model(x)
            emb = out["embedding"]
            logits = classifier(emb)
            loss_ce = ce_loss(logits, y)
            loss_tri = tri_loss(emb, y)
            loss = loss_ce + loss_tri
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        running["ce"] += loss_ce.item() * x.size(0)
        running["tri"] += loss_tri.item() * x.size(0)
    n = len(loader.dataset)
    return {k: v / n for k, v in running.items()}


def evaluate(model, classifier, ce_loss, loader, device):
    model.eval(); classifier.eval()
    running_ce = 0.0; correct = 0; total = 0
    with torch.no_grad():
        for x, y, view_id in loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            out = model(x)
            emb = out["embedding"]
            logits = classifier(emb)
            running_ce += ce_loss(logits, y).item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)
    return {"ce": running_ce / max(1, total), "acc": correct / max(1, total)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ddp", action="store_true", help="enable DDP; launch with torchrun")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MultiViewGaitDataset([Path(p) for p in args.data_roots], seq_len=args.seq_len, height=args.height, width=args.width)
    n_train = int(len(dataset) * 0.9)
    n_val = max(1, len(dataset) - n_train)
    train_set, val_set = random_split(dataset, [n_train, n_val])

    # Multi-view PK sampler
    train_labels = []; train_views = []
    for i in range(len(train_set)):
        _, y, v = train_set[i]
        train_labels.append(int(y)); train_views.append(int(v))
    sampler = MultiViewPKSampler(labels=train_labels, view_ids=train_views, batch_p=args.batch_p, batch_k=args.batch_k, views_per_id=args.views_per_id, balance_across_views=getattr(args, 'balance_across_views', True))
    # In DDP case, sampler will shard batches internally; here we consume lists of indices per batch
    train_loader = DataLoader(train_set, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_set, batch_size=args.batch_k * args.batch_p, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    model = LSTCBackbone(in_channels=1, base_channels=args.base_channels, num_stripes=args.num_stripes, embedding_dim=args.embedding_dim).to(device)
    num_classes = len(dataset.global_label_map)
    classifier = nn.Linear(args.embedding_dim, num_classes).to(device)

    ce_loss = nn.CrossEntropyLoss()
    tri_loss = BatchHardTripletLoss(margin=0.3, metric="euclidean")
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    tb_writer = create_tb_writer(log_dir / "tb", enabled=(args.tensorboard and (not args.ddp or dist.get_rank() == 0)))
    csv_logger = CSVLogger(log_dir / "train.csv", ["epoch", "train_ce", "train_tri", "val_ce", "val_acc", "lr"]) if (args.csv_log and (not args.ddp or dist.get_rank() == 0)) else None

    start_epoch = 1; best = {"acc": 0.0}
    ckpt = try_load_checkpoint(args.resume)
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        classifier.load_state_dict(ckpt["classifier"])
        if "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])  # type: ignore[arg-type]
        if "scheduler" in ckpt: scheduler.load_state_dict(ckpt["scheduler"])  # type: ignore[arg-type]
        start_epoch = ckpt.get("epoch", 0) + 1
        if "best" in ckpt: best = ckpt["best"]

    for epoch in range(start_epoch, args.epochs + 1):
        tr = train_epoch(model, classifier, ce_loss, tri_loss, train_loader, optimizer, device, amp=args.amp)
        ev = evaluate(model, classifier, ce_loss, val_loader, device)
        scheduler.step()
        if (not args.ddp) or dist.get_rank() == 0:
            print(f"Epoch {epoch:03d} | train ce={tr['ce']:.4f} tri={tr['tri']:.4f} | val ce={ev['ce']:.4f} acc={ev['acc']:.3f}")
        if tb_writer:
            tb_writer.add_scalar("train/ce", tr['ce'], epoch); tb_writer.add_scalar("train/tri", tr['tri'], epoch)
            tb_writer.add_scalar("val/ce", ev['ce'], epoch); tb_writer.add_scalar("val/acc", ev['acc'], epoch)
            tb_writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
        if csv_logger:
            csv_logger.log({"epoch": epoch, "train_ce": tr['ce'], "train_tri": tr['tri'], "val_ce": ev['ce'], "val_acc": ev['acc'], "lr": optimizer.param_groups[0]['lr']})

        state = {"model": model.state_dict(), "classifier": classifier.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch, "best": best, "args": vars(args), "num_classes": num_classes}
        if (not args.ddp) or dist.get_rank() == 0:
            save_checkpoint(state, out_dir / "last.pt")
            if ev["acc"] >= best["acc"]:
                best = ev
                save_checkpoint(state, out_dir / "best.pt")

    if tb_writer: tb_writer.close()
    if csv_logger: csv_logger.close()
    if (not args.ddp) or dist.get_rank() == 0:
        print(f"Best acc: {best['acc']:.3f}")
