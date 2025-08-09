from __future__ import annotations

import argparse
import sys
import subprocess
from pathlib import Path
from typing import List

import torch

from .model import LSTCBackbone
from torch import nn


def _run_script(script: str, argv: List[str]) -> int:
    script_path = str(Path(__file__).resolve().parent.parent / "examples" / script)
    return subprocess.call([sys.executable, script_path] + argv)


def cmd_sanity(args: argparse.Namespace) -> int:
    return _run_script("sanity_check.py", [])


def cmd_train_real(args: argparse.Namespace) -> int:
    return _run_script("train_real.py", args.passthrough)


def cmd_train_metric(args: argparse.Namespace) -> int:
    return _run_script("train_metric.py", args.passthrough)


def cmd_train_real_mv(args: argparse.Namespace) -> int:
    return _run_script("train_real_multiview.py", args.passthrough)


def cmd_train_metric_mv(args: argparse.Namespace) -> int:
    return _run_script("train_metric_multiview.py", args.passthrough)


def cmd_eval(args: argparse.Namespace) -> int:
    return _run_script("eval_retrieval.py", args.passthrough)


def cmd_eval_mv(args: argparse.Namespace) -> int:
    return _run_script("eval_retrieval_multiview.py", args.passthrough)


def cmd_export(args: argparse.Namespace) -> int:
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 2
    ckpt = torch.load(ckpt_path, map_location="cpu")
    embed_dim = ckpt.get("args", {}).get("embedding_dim", 256)
    num_stripes = ckpt.get("args", {}).get("num_stripes", 8)
    base_channels = ckpt.get("args", {}).get("base_channels", 16)
    model = LSTCBackbone(in_channels=1, base_channels=base_channels, num_stripes=num_stripes, embedding_dim=embed_dim)
    model.load_state_dict(ckpt["model"])  # type: ignore[index]
    model.eval()

    # TorchScript export (trace on dummy input)
    t, h, w = args.seq_len, args.height, args.width
    dummy = torch.randn(1, 1, t, h, w)
    scripted = torch.jit.trace(lambda x: model(x)["embedding"], dummy)
    ts_out = Path(args.torchscript_out)
    ts_out.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(ts_out))
    print(f"Saved TorchScript to {ts_out}")

    # Optional ONNX export
    if args.onnx_out:
        onnx_out = Path(args.onnx_out)
        onnx_out.parent.mkdir(parents=True, exist_ok=True)
        try:
            class _EmbeddingOnly(nn.Module):
                def __init__(self, m: nn.Module) -> None:
                    super().__init__()
                    self.m = m

                def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                    return self.m(x)["embedding"]

            torch.onnx.export(
                _EmbeddingOnly(model),
                dummy,
                str(onnx_out),
                export_params=True,
                input_names=["input"],
                output_names=["embedding"],
                dynamic_axes={"input": {0: "N", 2: "T"}, "embedding": {0: "N"}},
                opset_version=args.opset,
            )
            print(f"Saved ONNX to {onnx_out}")
        except Exception as e:  # noqa: BLE001
            print(f"ONNX export failed: {e}", file=sys.stderr)
            return 3
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lstc")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # passthrough-style subcommands to reuse example scripts unmodified
    for name, func, help_text in [
        ("sanity", cmd_sanity, "Run quick shape sanity-check"),
        ("train-real", cmd_train_real, "Train on real dataset (CE)"),
        ("train-metric", cmd_train_metric, "Train with CE+Triplet (PK)"),
        ("train-real-mv", cmd_train_real_mv, "Train CE (multi-view)"),
        ("train-metric-mv", cmd_train_metric_mv, "Train CE+Triplet (multi-view PK)"),
        ("eval", cmd_eval, "Evaluate retrieval (single-view)"),
        ("eval-mv", cmd_eval_mv, "Evaluate retrieval (multi-view)")
    ]:
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(_func=func)
        p.add_argument("passthrough", nargs=argparse.REMAINDER, help="Arguments forwarded to the underlying script")

    # export
    pexp = sub.add_parser("export", help="Export checkpoint to TorchScript and optionally ONNX")
    pexp.add_argument("--ckpt", required=True, help="Path to checkpoint (best.pt)")
    pexp.add_argument("--seq-len", type=int, default=30)
    pexp.add_argument("--height", type=int, default=64)
    pexp.add_argument("--width", type=int, default=44)
    pexp.add_argument("--torchscript-out", type=str, default="runs/export/model.ts")
    pexp.add_argument("--onnx-out", type=str, default="", help="Optional ONNX output path")
    pexp.add_argument("--opset", type=int, default=17)
    pexp.set_defaults(_func=cmd_export)

    args = parser.parse_args(argv)
    return int(args._func(args))


if __name__ == "__main__":
    raise SystemExit(main())
