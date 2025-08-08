import argparse
import itertools
import subprocess
import sys
from pathlib import Path


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
    ap.add_argument("--out", type=str, default="runs/ablation")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    toggles = [
        (True, True, True),
        (True, False, False),  # temporal only
        (False, True, False),  # spatial only
        (False, False, True),  # joint only
        (True, True, False),
        (True, False, True),
        (False, True, True),
    ]

    for use_temporal, use_spatial, use_joint in toggles:
        tag = f"t{int(use_temporal)}_s{int(use_spatial)}_j{int(use_joint)}"
        out = out_dir / tag
        cmd = [
            sys.executable, "examples/train_real.py",
            "--data-root", args.data_root,
            "--seq-len", str(args.seq_len),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--device", args.device,
            "--use-temporal" if use_temporal else "",
            "--use-spatial" if use_spatial else "",
            "--use-joint" if use_joint else "",
            "--out-dir", str(out),
        ]
        cmd = [c for c in cmd if c != ""]
        run(cmd)

    print(f"Ablation runs finished: outputs under {out_dir}")


if __name__ == "__main__":
    main()
