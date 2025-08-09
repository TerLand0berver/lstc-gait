import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from lstc import LSTCBackbone
from examples.dataset_silhouette import scan_silhouette_root, GaitSilhouetteDataset


def compute_embeddings(model, loader, device):
    model.eval()
    feats = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, y, idx = batch
            else:
                x, y = batch
            x = x.to(device, non_blocking=True)
            out = model(x)
            emb = out["embedding"].detach().cpu().numpy()
            feats.append(emb)
            labels.append(y.numpy())
    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)
    # L2 normalize
    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
    feats = feats / norms
    return feats, labels


def cmc_map(query_feats, query_labels, gallery_feats, gallery_labels, ranks=(1, 5, 10)):
    # Cosine similarity (since normalized)
    sims = query_feats @ gallery_feats.T  # (Nq, Ng)
    order = np.argsort(-sims, axis=1)

    # Exclude self-matches when query and gallery are the same set
    same_set = query_feats.shape[0] == gallery_feats.shape[0] and np.all(query_labels == gallery_labels)

    cmc = np.zeros(max(ranks))
    ap_list = []
    for i in range(order.shape[0]):
        gt = query_labels[i]
        ranking = order[i]
        if same_set:
            # remove self index if appears at top
            # We do not have original indices; approximate by skipping first if label equal and sim is 1.0.
            if np.isclose(sims[i, ranking[0]], 1.0) and gallery_labels[ranking[0]] == gt:
                ranking = ranking[1:]
        good = (gallery_labels[ranking] == gt).astype(np.int32)
        # CMC
        first_hit = np.argmax(good)
        for r in range(max(ranks)):
            if first_hit <= r:
                cmc[r] += 1
        # AP
        if good.sum() == 0:
            ap_list.append(0.0)
        else:
            hits = 0
            precisions = []
            for rank, g in enumerate(good, start=1):
                if g == 1:
                    hits += 1
                    precisions.append(hits / rank)
            ap_list.append(np.mean(precisions))
    cmc = cmc / order.shape[0]
    mAP = float(np.mean(ap_list))
    return {f"CMC@{r}": cmc[r-1] for r in ranks} | {"mAP": mAP}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=44)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    records, label_map = scan_silhouette_root(Path(args.data_root), min_frames=2)
    dataset = GaitSilhouetteDataset(records, seq_len=args.seq_len, height=args.height, width=args.width, sampling="uniform", return_index=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    ckpt = torch.load(args.ckpt, map_location="cpu")
    embed_dim = ckpt.get("args", {}).get("embedding_dim", 256)
    num_stripes = ckpt.get("args", {}).get("num_stripes", 8)
    base_channels = ckpt.get("args", {}).get("base_channels", 16)

    model = LSTCBackbone(in_channels=1, base_channels=base_channels, num_stripes=num_stripes, embedding_dim=embed_dim)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    feats, labels = compute_embeddings(model, loader, device)
    metrics = cmc_map(feats, labels, feats, labels, ranks=(1, 5, 10))
    print({k: float(v) for k, v in metrics.items()})
