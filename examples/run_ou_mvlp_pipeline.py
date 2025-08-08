import argparse
import subprocess
from pathlib import Path
import sys


def run(cmd: list[str]):
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--views", type=str, default="000,015,030,045,060,075,090,180")
    ap.add_argument("--seq-len", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out-dir", type=str, default="runs/ou_mvlp_pipeline")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "best.pt"

    # Train
    train_cmd = [
        sys.executable, "examples/train_ou_mvlp.py",
        "--data-root", args.data_root,
        "--views", args.views,
        "--seq-len", str(args.seq_len),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--device", args.device,
        "--out-dir", str(out),
    ]
    run(train_cmd)

    # Eval (self-gallery)
    eval_cmd = [
        sys.executable, "examples/eval_ou_mvlp.py",
        "--data-root", args.data_root,
        "--ckpt", str(ckpt),
        "--views", args.views,
        "--seq-len", str(args.seq_len),
    ]
    run(eval_cmd)

    print(f"Pipeline done. Best ckpt at: {ckpt}")


if __name__ == "__main__":
    main()
