import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
import torch
import csv


BRANCH_MODES = {
    "tsj": (True, True, True),
    "t--": (True, False, False),
    "-s-": (False, True, False),
    "--j": (False, False, True),
    "ts-": (True, True, False),
    "t-j": (True, False, True),
    "-sj": (False, True, True),
}


def run(cmd: list[str]):
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--seq-len", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="runs/sweep")
    ap.add_argument("--stripes", type=str, default="6,8,10")
    ap.add_argument("--topks", type=str, default="2,3,4")
    ap.add_argument("--modes", type=str, default="tsj,t--,-s-,--j,ts-,t-j,-sj")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    stripes = [int(x) for x in args.stripes.split(",") if x.strip()]
    topks = [int(x) for x in args.topks.split(",") if x.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    rows = []
    for s in stripes:
        for k in topks:
            for mode in modes:
                if mode not in BRANCH_MODES:
                    print(f"[WARN] unknown mode {mode}, skip")
                    continue
                use_temporal, use_spatial, use_joint = BRANCH_MODES[mode]
                tag = f"S{s}_K{k}_{mode}"
                out = out_dir / tag
                out.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable, "examples/train_real.py",
                    "--data-root", args.data_root,
                    "--seq-len", str(args.seq_len),
                    "--epochs", str(args.epochs),
                    "--batch-size", str(args.batch_size),
                    "--device", args.device,
                    "--num-stripes", str(s),
                    "--pooling-topk", str(k),
                    "--out-dir", str(out),
                ]
                if use_temporal:
                    cmd.append("--use-temporal")
                if use_spatial:
                    cmd.append("--use-spatial")
                if use_joint:
                    cmd.append("--use-joint")
                run(cmd)
                ckpt_path = out / "best.pt"
                if ckpt_path.exists():
                    ckpt = torch.load(ckpt_path, map_location="cpu")
                    val_acc = float(ckpt.get("val_acc", 0.0))
                else:
                    val_acc = float("nan")
                rows.append({"mode": mode, "num_stripes": s, "topk": k, "val_acc": val_acc})

    # Save CSV
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "num_stripes", "topk", "val_acc"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    # Save Markdown table
    md_path = out_dir / "summary.md"
    with open(md_path, "w") as f:
        f.write("| mode | S | top-k | val_acc |\n")
        f.write("|:----:|---:|-----:|-------:|\n")
        for r in rows:
            f.write(f"| {r['mode']} | {r['num_stripes']} | {r['topk']} | {r['val_acc']:.4f} |\n")
    print(f"Sweep done. Summary at {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
