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
    ap.add_argument("--per-view", action="store_true", help="report metrics per probe view and overall average")
    ap.add_argument("--export-csv", type=str, default="", help="optional CSV path to save per-view metrics")
    ap.add_argument("--export-md", type=str, default="", help="optional Markdown path to save per-view table")
    ap.add_argument("--preset", type=str, default="", choices=["", "casia-b-standard"], help="use a predefined protocol; overrides filters")
    ap.add_argument("--seq-len", type=int, default=30)
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--width", type=int, default=44)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # Apply preset if requested
    if args.preset == "casia-b-standard":
        args.views = args.views or "000,018,036,054,072,090,108,126,144,162,180"
        args.gallery_conds = "nm"
        args.gallery_cond_ids = "01,02,03,04"
        args.probe_conds = "nm,bg,cl"
        args.probe_cond_ids = "05,06,01,02,01,02"

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

    def compute_metrics(mask: np.ndarray):
        return cmc_map_masked(pr_feats, pr_labels, gal_feats, gal_labels, mask, ranks=(1,5,10))

    results = {}
    rows = []
    if args.per_view:
        unique_pr_views = np.unique(pr_views).tolist()
        for v in unique_pr_views:
            if args.cross_view:
                mask = (pr_views[:, None] == v) & (gal_views[None, :] != v)
            else:
                mask = (pr_views[:, None] == v) & (gal_views[None, :] == v)
            m = compute_metrics(mask)
            results[str(v)] = m
            rows.append({"view": int(v), **{k: float(val) for k, val in m.items()}})
        # overall average across probe views
        avg = {k: float(np.mean([results[str(v)][k] for v in unique_pr_views])) for k in next(iter(results.values())).keys()}
        results["avg"] = avg
        rows.append({"view": "avg", **avg})
        print({"per_view": results})
        # export CSV if requested
        if args.export_csv:
            import csv
            from pathlib import Path as _P
            outp = _P(args.export_csv)
            outp.parent.mkdir(parents=True, exist_ok=True)
            with open(outp, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["view", "CMC@1", "CMC@5", "CMC@10", "mAP"])
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
        # export Markdown if requested
        if args.export_md:
            from pathlib import Path as _P
            outp = _P(args.export_md)
            outp.parent.mkdir(parents=True, exist_ok=True)
            with open(outp, "w") as f:
                f.write("| view | CMC@1 | CMC@5 | CMC@10 | mAP |\n")
                f.write("|---:|---:|---:|---:|---:|\n")
                for r in rows:
                    f.write(f"| {r['view']} | {r['CMC@1']:.3f} | {r['CMC@5']:.3f} | {r['CMC@10']:.3f} | {r['mAP']:.3f} |\n")
    else:
        if args.cross_view:
            mask = pr_views[:, None] != gal_views[None, :]
            metrics = compute_metrics(mask)
        else:
            mask = pr_views[:, None] == gal_views[None, :]
            metrics = compute_metrics(mask)
        print({k: float(v) for k, v in metrics.items()})


if __name__ == "__main__":
    main()
