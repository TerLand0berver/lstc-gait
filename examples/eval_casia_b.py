import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from lstc import LSTCBackbone
from examples.dataset_casia_b import scan_casia_b, build_casia_b_dataset, CasiaBRecord


def cmc_map(query_feats, query_labels, gallery_feats, gallery_labels, ranks=(1, 5, 10)):
    sims = query_feats @ gallery_feats.T
    order = np.argsort(-sims, axis=1)
    cmc = np.zeros(max(ranks)); ap_list = []
    for i in range(order.shape[0]):
        gt = query_labels[i]
        ranking = order[i]
        good = (gallery_labels[ranking] == gt).astype(np.int32)
        if good.sum() == 0:
            ap_list.append(0.0); continue
        first_hit = np.argmax(good)
        for r in range(max(ranks)):
            if first_hit <= r:
                cmc[r] += 1
        hits=0; precisions=[]
        for rank, g in enumerate(good, start=1):
            if g==1:
                hits += 1; precisions.append(hits/rank)
        ap_list.append(np.mean(precisions))
    cmc = cmc / order.shape[0]
    mAP = float(np.mean(ap_list))
    return {f"CMC@{r}": cmc[r-1] for r in ranks} | {"mAP": mAP}


def cmc_map_masked(query_feats, query_labels, gallery_feats, gallery_labels, valid_mask: np.ndarray, ranks=(1, 5, 10)):
    # valid_mask: (Nq, Ng) boolean; True means allowed match
    sims = query_feats @ gallery_feats.T
    sims[~valid_mask] = -1e9
    order = np.argsort(-sims, axis=1)
    cmc = np.zeros(max(ranks)); ap_list = []
    for i in range(order.shape[0]):
        gt = query_labels[i]
        ranking = order[i]
        good = (gallery_labels[ranking] == gt).astype(np.int32)
        if good.sum() == 0:
            ap_list.append(0.0); continue
        first_hit = np.argmax(good)
        for r in range(max(ranks)):
            if first_hit <= r:
                cmc[r] += 1
        hits=0; precisions=[]
        for rank, g in enumerate(good, start=1):
            if g==1:
                hits += 1; precisions.append(hits/rank)
        ap_list.append(np.mean(precisions))
    cmc = cmc / order.shape[0]
    mAP = float(np.mean(ap_list))
    return {f"CMC@{r}": cmc[r-1] for r in ranks} | {"mAP": mAP}


def _select_records(records: list[CasiaBRecord], conds: list[str], cond_ids: list[str], views: list[str] | None) -> list[CasiaBRecord]:
    out = []
    for r in records:
        if r.condition not in conds:
            continue
        if cond_ids and r.cond_id not in cond_ids:
            continue
        if views and r.view not in views:
            continue
        out.append(r)
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--views", type=str, default="")
    ap.add_argument("--gallery-conds", type=str, default="nm", help="e.g., nm")
    ap.add_argument("--gallery-cond-ids", type=str, default="01,02,03,04", help="e.g., 01,02,03,04")
    ap.add_argument("--probe-conds", type=str, default="nm,bg,cl", help="e.g., nm,bg,cl")
    ap.add_argument("--probe-cond-ids", type=str, default="05,06,01,02,01,02", help="nm:05,06; bg:01,02; cl:01,02")
    ap.add_argument("--cross-view", action="store_true", help="evaluate cross-view (exclude same view)")
    ap.add_argument("--seq-len", type=int, default=30)
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--width", type=int, default=44)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    views = [v.strip() for v in args.views.split(",") if v.strip()] if args.views else None

    all_records = scan_casia_b(Path(args.data_root), include_views=views, include_conditions=None, min_frames=2)

    gal_conds = [c.strip() for c in args.gallery_conds.split(",") if c.strip()]
    gal_ids = [i.strip() for i in args.gallery_cond_ids.split(",") if i.strip()]
    pr_conds = [c.strip() for c in args.probe_conds.split(",") if c.strip()]
    pr_ids = [i.strip() for i in args.probe_cond_ids.split(",") if i.strip()]

    gal_records = _select_records(all_records, gal_conds, gal_ids, views)
    pr_records = _select_records(all_records, pr_conds, pr_ids, views)

    if len(gal_records) == 0 or len(pr_records) == 0:
        raise SystemExit("Empty gallery or probe after filtering; check flags")

    gal_ds = build_casia_b_dataset(gal_records, seq_len=args.seq_len, height=args.height, width=args.width)
    pr_ds = build_casia_b_dataset(pr_records, seq_len=args.seq_len, height=args.height, width=args.width)
    gal_loader = DataLoader(gal_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    pr_loader = DataLoader(pr_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    ckpt = torch.load(Path(args.ckpt), map_location="cpu")
    embed_dim = ckpt.get("args", {}).get("embedding_dim", 256)
    num_stripes = ckpt.get("args", {}).get("num_stripes", 8)
    base_channels = ckpt.get("args", {}).get("base_channels", 16)

    device = torch.device(args.device)
    model = LSTCBackbone(in_channels=1, base_channels=base_channels, num_stripes=num_stripes, embedding_dim=embed_dim)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    def embed(loader):
        feats=[]; labels=[]
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            emb = out["embedding"].detach().cpu().numpy()
            feats.append(emb); labels.append(y.numpy())
        feats = np.concatenate(feats, axis=0)
        labels = np.concatenate(labels, axis=0)
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
        return feats, labels

    gal_feats, gal_labels = embed(gal_loader)
    pr_feats, pr_labels = embed(pr_loader)

    # Build view arrays aligned with dataset order (shuffle=False preserves record order)
    gal_views = np.array([int(r.view) for r in gal_records], dtype=np.int32)
    pr_views = np.array([int(r.view) for r in pr_records], dtype=np.int32)

    if args.cross_view:
        mask = pr_views[:, None] != gal_views[None, :]
        metrics = cmc_map_masked(pr_feats, pr_labels, gal_feats, gal_labels, mask, ranks=(1,5,10))
    else:
        metrics = cmc_map(pr_feats, pr_labels, gal_feats, gal_labels)
    print({k: float(v) for k, v in metrics.items()})


if __name__ == "__main__":
    main()
