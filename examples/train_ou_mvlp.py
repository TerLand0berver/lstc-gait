import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from lstc import LSTCBackbone
from examples.dataset_ou_mvlp import scan_ou_mvlp, build_ou_dataset
from lstc.utils import save_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--views", type=str, default="")
    ap.add_argument("--seq-len", type=int, default=30)
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--width", type=int, default=44)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--base-channels", type=int, default=16)
    ap.add_argument("--num-stripes", type=int, default=8)
    ap.add_argument("--embedding-dim", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", type=str, default="runs/ou_mvlp")
    args = ap.parse_args()

    views = [v.strip() for v in args.views.split(",") if v.strip()] if args.views else None

    recs = scan_ou_mvlp(Path(args.data_root), include_views=views, min_frames=2)
    if len(recs) < 10:
        raise SystemExit("Too few sequences; check --data-root or filters")

    ds = build_ou_dataset(recs, seq_len=args.seq_len, height=args.height, width=args.width)
    n_train = int(0.9 * len(ds)); n_val = max(1, len(ds) - n_train)
    train_set, val_set = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device(args.device)
    model = LSTCBackbone(in_channels=1, base_channels=args.base_channels, num_stripes=args.num_stripes, embedding_dim=args.embedding_dim).to(device)
    num_classes = max(r.label_id for r in recs) + 1
    classifier = nn.Linear(args.embedding_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = 0.0
    for epoch in range(1, args.epochs + 1):
        # train
        model.train(); classifier.train()
        loss_sum = 0.0; correct = 0; total = 0
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            optim.zero_grad(set_to_none=True)
            emb = model(x)["embedding"]
            logits = classifier(emb)
            loss = criterion(logits, y)
            loss.backward(); optim.step()
            loss_sum += loss.item() * x.size(0)
            correct += (logits.detach().argmax(dim=1) == y).sum().item()
            total += x.size(0)
        tr_loss, tr_acc = loss_sum / max(1, total), correct / max(1, total)

        # val
        model.eval(); classifier.eval(); loss_sum = 0.0; correct = 0; total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device); y = y.to(device)
                emb = model(x)["embedding"]
                logits = classifier(emb)
                loss = criterion(logits, y)
                loss_sum += loss.item() * x.size(0)
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += x.size(0)
        va_loss, va_acc = loss_sum / max(1, total), correct / max(1, total)
        sched.step()
        print(f"Epoch {epoch}: train loss={tr_loss:.4f} acc={tr_acc:.3f} | val loss={va_loss:.4f} acc={va_acc:.3f}")
        # save last
        state = {
            "model": model.state_dict(),
            "classifier": classifier.state_dict(),
            "epoch": epoch,
            "val_acc": va_acc,
            "args": vars(args),
        }
        save_checkpoint(state, out_dir / "last.pt")
        if va_acc >= best:
            best = va_acc
            save_checkpoint(state, out_dir / "best.pt")

    print(f"Best val acc: {best:.3f}")


if __name__ == "__main__":
    main()
