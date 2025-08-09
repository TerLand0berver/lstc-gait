import argparse
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader

from lstc import LSTCBackbone
from examples.dataset_silhouette import scan_silhouette_root, GaitSilhouetteDataset


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (best.pt)")
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=44)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default="runs/embeddings.jsonl")
    args = parser.parse_args()

    device = torch.device(args.device)
    records, label_map = scan_silhouette_root(Path(args.data_root), min_frames=2)
    dataset = GaitSilhouetteDataset(records, seq_len=args.seq_len, height=args.height, width=args.width, sampling="uniform")
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    ckpt = torch.load(args.ckpt, map_location="cpu")
    embed_dim = ckpt.get("args", {}).get("embedding_dim", 256)
    num_stripes = ckpt.get("args", {}).get("num_stripes", 8)
    base_channels = ckpt.get("args", {}).get("base_channels", 16)

    model = LSTCBackbone(in_channels=1, base_channels=base_channels, num_stripes=num_stripes, embedding_dim=embed_dim)
    model.load_state_dict(ckpt["model"])
    model.to(device); model.eval()

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            out = model(x)
            emb = out["embedding"].detach().cpu().numpy()
            for i in range(emb.shape[0]):
                row = {"label": int(y[i]), "embedding": emb[i].tolist()}
                f.write(json.dumps(row) + "\n")

    print(f"Saved embeddings to {out_path}")


if __name__ == "__main__":
    main()
