import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from lstc import LSTCBackbone
from examples.dataset_ou_mvlp import scan_ou_mvlp, build_ou_dataset
import json


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


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--views", type=str, default="")
    ap.add_argument("--seq-len", type=int, default=30)
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--width", type=int, default=44)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    views = [v.strip() for v in args.views.split(",") if v.strip()] if args.views else None
    records = scan_ou_mvlp(Path(args.data_root), include_views=views, min_frames=2)
    dataset = build_ou_dataset(records, seq_len=args.seq_len, height=args.height, width=args.width)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    ckpt = torch.load(Path(args.ckpt), map_location="cpu")
    embed_dim = ckpt.get("args", {}).get("embedding_dim", 256)
    num_stripes = ckpt.get("args", {}).get("num_stripes", 8)
    base_channels = ckpt.get("args", {}).get("base_channels", 16)

    device = torch.device(args.device)
    model = LSTCBackbone(in_channels=1, base_channels=base_channels, num_stripes=num_stripes, embedding_dim=embed_dim)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    feats=[]; labels=[]
    for x, y in loader:
        x = x.to(device)
        emb = model(x)["embedding"].detach().cpu().numpy()
        feats.append(emb); labels.append(y.numpy())
    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)

    metrics = cmc_map(feats, labels, feats, labels)
    out = {k: float(v) for k, v in metrics.items()}
    if args.json:
        print(json.dumps(out))
    else:
        print(out)


if __name__ == "__main__":
    main()
