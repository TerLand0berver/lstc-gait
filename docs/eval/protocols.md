# Evaluation Protocols

This document describes the evaluation protocols used by the example scripts.

## CASIA-B
- Standard preset (used by `--preset casia-b-standard` in `examples/eval_casia_b.py`):
  - Gallery: nm (normal) condition with sequence ids 01–04
  - Probe: nm with 05–06, bg with 01–02, cl with 01–02
  - Views: user-specified or the common 11 views: 000,018,036,054,072,090,108,126,144,162,180
  - Reporting:
    - Per-probe-view CMC@m and mAP, plus macro average across probe views
    - Cross-view mode (`--cross-view`) excludes same-view matches
  - Export:
    - CSV via `--export-csv`
    - Markdown via `--export-md`

Command example:
```
uv run python examples/eval_casia_b.py \
  --data-root /path/to/CASIA-B --ckpt /path/to/best.pt \
  --preset casia-b-standard --per-view --cross-view \
  --export-csv runs/casia_b_eval.csv --export-md runs/casia_b_eval.md
```

## OU-MVLP
- The example provides a self-gallery retrieval evaluation over selected views (e.g., 000,015,...,180) and reports CMC@m and mAP.
- Protocol variations (e.g., cross-view only, per-view reporting) can be extended analogously to CASIA-B.

Command example:
```
uv run python examples/eval_ou_mvlp.py \
  --data-root /path/to/OU-MVLP --ckpt runs/ou_mvlp/best.pt \
  --views 000,015,030,045,060,075,090,180
```
