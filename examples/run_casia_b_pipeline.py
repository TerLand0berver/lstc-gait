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
    ap.add_argument("--views", type=str, default="000,018,036,054,072,090,108,126,144,162,180")
    ap.add_argument("--seq-len", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out-dir", type=str, default="runs/casia_b_pipeline")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "best.pt"
    ckpt_ema = out / "best_ema.pt"
    csv = out / "eval_per_view.csv"

    # 1) Train CE
    train_cmd = [
        sys.executable, "examples/train_casia_b.py",
        "--data-root", args.data_root,
        "--views", args.views,
        "--seq-len", str(args.seq_len),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--device", args.device,
        "--ema",
        "--out-dir", str(out),
    ]
    run(train_cmd)

    # 2) Evaluate per-view (cross-view), export CSV
    # Prefer EMA checkpoint if available
    ckpt_to_use = ckpt_ema if ckpt_ema.exists() else ckpt

    eval_cmd = [
        sys.executable, "examples/eval_casia_b.py",
        "--data-root", args.data_root,
        "--ckpt", str(ckpt_to_use),
        "--views", args.views,
        "--gallery-conds", "nm",
        "--gallery-cond-ids", "01,02,03,04",
        "--probe-conds", "nm,bg,cl",
        "--probe-cond-ids", "05,06,01,02,01,02",
        "--per-view", "--cross-view", "--preset", "casia-b-standard",
        "--export-csv", str(csv), "--export-md", str(out / "eval_per_view.md"),
        "--seq-len", str(args.seq_len),
    ]
    run(eval_cmd)

    print(f"Pipeline done. Results at: {csv}")


if __name__ == "__main__":
    main()
