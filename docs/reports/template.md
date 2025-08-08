# Experiment Report Template

- Model: LSTCBackbone (base_channels=?, stripes=S, pooling_topk=K)
- Dataset: [CASIA-B | OU-MVLP]
- Training: epochs=?, batch-size=?, optimizer=?, lr=?, wd=?, EMA=?, grad-clip=?

## Results

### Per-view (if applicable)
- See exported CSV/MD from evaluation scripts

### Ablations (branch toggles / S / top-k)
- Summary table (paste from runs/sweep/summary.md):

| mode | S | top-k | val_acc |
|:----:|---:|-----:|-------:|
| tsj | 8 | 2 | 0.XXX |
| t-- | 8 | 2 | 0.XXX |
| -s- | 8 | 2 | 0.XXX |
| --j | 8 | 2 | 0.XXX |

## Observations
- ...

## Notes
- Random seed / hardware / runtime
