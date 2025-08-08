import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

from lstc import LSTCBackbone
from examples.dataset_multiview import MultiViewGaitDataset
from omegaconf import OmegaConf


def compute_feats(model, loader, device):
    model.eval()
    feats, labels, views = [], [], []
    with torch.no_grad():
        for x, y, view_id in loader:
            x = x.to(device, non_blocking=True)
            out = model(x)
            emb = out["embedding"].detach().cpu().numpy()
            feats.append(emb); labels.append(y.numpy()); views.append(view_id.numpy())
    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)
    views = np.concatenate(views, axis=0)
    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
    feats = feats / norms
    return feats, labels, views


def cmc_map_cross_view(query_feats, query_labels, query_views, gallery_feats, gallery_labels, gallery_views, ranks=(1,5,10)):
    # Only consider cross-view matches: different view between query and gallery
    mask = query_views[:, None] != gallery_views[None, :]
    sims = query_feats @ gallery_feats.T
    sims[~mask] = -1e9
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--ddp", action="store_true", help="gather embeddings across ranks")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cli = OmegaConf.create(vars(args))
    cfg = OmegaConf.merge(cfg, cli)  # ensure CLI (e.g., --ckpt) is preserved/overrides
    args = argparse.Namespace(**OmegaConf.to_container(cfg, resolve=True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MultiViewGaitDataset([Path(p) for p in args.data_roots], seq_len=args.seq_len, height=args.height, width=args.width)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    ckpt = torch.load(Path(args.ckpt), map_location="cpu")
    embed_dim = ckpt.get("args", {}).get("embedding_dim", args.embedding_dim)
    num_stripes = ckpt.get("args", {}).get("num_stripes", args.num_stripes)
    base_channels = ckpt.get("args", {}).get("base_channels", args.base_channels)
    model = LSTCBackbone(in_channels=1, base_channels=base_channels, num_stripes=num_stripes, embedding_dim=embed_dim)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    feats, labels, views = compute_feats(model, loader, device)
    # All-gather across ranks if DDP
    if args.ddp and dist.is_available() and dist.is_initialized():
        def allgather_numpy(arr: np.ndarray) -> np.ndarray:
            t = torch.from_numpy(arr).to(device)
            parts = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
            dist.all_gather(parts, t)
            return torch.cat(parts, dim=0).cpu().numpy()
        feats = allgather_numpy(feats)
        labels = allgather_numpy(labels.astype(np.int64)).astype(np.int64)
        views = allgather_numpy(views.astype(np.int64)).astype(np.int64)
    metrics = cmc_map_cross_view(feats, labels, views, feats, labels, views, ranks=(1,5,10))
    if (not args.ddp) or dist.get_rank() == 0:
        print({k: float(v) for k,v in metrics.items()})
