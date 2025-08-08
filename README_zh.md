## LSTC/LSTP 步态识别原型

在高度条带先验下学习判别性的步态时空特征：
- LSTC：在高度条带内进行 3D 深度卷积 + 1x1x1 通道混合，避免跨条带泄漏。
- 非对称块：时间分支(kT,1,1) / 空间分支(1,kH,kW) / 联合分支(LSTC) 并联融合。
- LSTP：条带内按帧 L2 得分做 top-k 选择并平均，再拼接得到序列表示。

### 目录结构
- `lstc/modules.py`：LSTC、非对称块、LSTP
- `lstc/model.py`：参考骨干 `LSTCBackbone`
- `lstc/losses.py`、`lstc/samplers.py`、`lstc/utils.py`
- 示例：`examples/`（toy/real/metric/multiview 训练 | 评估 | 导出）
- 配置：`configs/real.yaml`、`configs/metric.yaml`、`configs/multiview_real.yaml`、`configs/multiview_metric.yaml`、`configs/casia_b.yaml`、`configs/ou_mvlp.yaml`

### 安装（推荐 uv + GPU）
1) 创建环境并安装 PyTorch（选择匹配的 CUDA 构建）：
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv -p 3.11 .venv && source .venv/bin/activate
# 例：CUDA 12.1（按需调整）
uv pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
```
2) 安装项目依赖（不含 torch）：
```bash
uv pip install -r requirements.txt --no-deps
```
3) 验证 GPU：
```bash
uv run python -c "import torch;print(torch.cuda.is_available(), torch.cuda.device_count())"
```

### 快速上手
- 形状健检：
```bash
uv run python examples/sanity_check.py
```
- 玩具训练：
```bash
uv run python examples/train_toy.py --epochs 2 --device cuda
```
- 真实数据（CE）+ 训练增强：
```bash
uv run python examples/train_real.py --data-root /path/to/data --epochs 50 --batch-size 32 --seq-len 30 --device cuda --amp \
  --ema --ema-decay 0.999 --grad-clip 1.0  # 保存 best_ema.pt/last_ema.pt，并打印 ema acc \
  --tensorboard --csv-log --log-dir runs/logs_real
```
- 真实数据（CE+Triplet，PK）：
```bash
uv run python examples/train_metric.py --data-root /path/to/data --epochs 50 --batch-p 8 --batch-k 4 --seq-len 30 --device cuda --amp \
  --ema --grad-clip 1.0  # 保存 best_ema.pt/last_ema.pt，并打印 ema acc
```

### 配置驱动运行
```bash
# CE
uv run python examples/train_real.py --config configs/real.yaml --device cuda
# CE+Triplet
uv run python examples/train_metric.py --config configs/metric.yaml --device cuda
```

### 多视角（跨视角）
- 多视角 CE 训练：
```bash
# 编辑 configs/multiview_real.yaml 的 data_roots
uv run python examples/train_real_multiview.py --config configs/multiview_real.yaml
```
- 多视角 CE+Triplet（MultiView PK 采样）：
```bash
uv run python examples/train_metric_multiview.py --config configs/multiview_metric.yaml
# 关键参数：batch_p, batch_k, views_per_id, balance_across_views
```
- 跨视角检索评估（CMC/mAP）：
```bash
uv run python examples/eval_retrieval_multiview.py --config configs/multiview_real.yaml \
  --ckpt runs/lstc_real_mv/best.pt
```

### CASIA-B
- 最小 CE 训练：
```bash
uv run python examples/train_casia_b.py \
  --data-root /path/to/CASIA-B --conds nm \
  --views 000,018,036,054,072,090,108,126,144,162,180 \
  --seq-len 30 --epochs 50 --batch-size 32 --device cuda
```
- 评估（gallery/probe 协议，逐视角 CSV/Markdown）：
```bash
# 使用预设协议并同时导出 CSV 与 Markdown 表格
uv run python examples/eval_casia_b.py \
  --data-root /path/to/CASIA-B --ckpt /path/to/best.pt \
  --preset casia-b-standard --per-view --cross-view \
  --export-csv runs/casia_b_eval.csv --export-md runs/casia_b_eval.md
```
- 一键流水线（训练 + 跨视角逐视角评估→CSV/MD，优先使用 EMA 检查点）：
```bash
uv run python examples/run_casia_b_pipeline.py \
  --data-root /path/to/CASIA-B \
  --views 000,018,036,054,072,090,108,126,144,162,180 \
  --seq-len 30 --epochs 50 --batch-size 32 --device cuda \
  --out-dir runs/casia_b_pipeline
```

### OU-MVLP
- 最小 CE 训练：
```bash
uv run python examples/train_ou_mvlp.py \
  --data-root /path/to/OU-MVLP \
  --views 000,015,030,045,060,075,090,180 \
  --seq-len 30 --epochs 50 --batch-size 64 --device cuda --out-dir runs/ou_mvlp
```
- 评估（自集合 CMC/mAP）：
```bash
uv run python examples/eval_ou_mvlp.py \
  --data-root /path/to/OU-MVLP --ckpt runs/ou_mvlp/best.pt \
  --views 000,015,030,045,060,075,090,180
```
- 一键流水线：
```bash
uv run python examples/run_ou_mvlp_pipeline.py \
  --data-root /path/to/OU-MVLP \
  --views 000,015,030,045,060,075,090,180 \
  --seq-len 30 --epochs 50 --batch-size 64 --device cuda \
  --out-dir runs/ou_mvlp_pipeline
```

### 分布式训练与评估（DDP）
- 使用 torchrun 启动；batch-size 为每卡大小。
- 使用（多视角）PK 采样的度量学习需满足 `P*K % world_size == 0`，采样器会将每个批次按 rank 均分。
```bash
# CE（单视角）
torchrun --nproc_per_node=4 examples/train_real.py --data-root /path/to/data \
  --epochs 50 --batch-size 16 --seq-len 30 --ddp --amp --tensorboard --csv-log
# 多视角度量学习
torchrun --nproc_per_node=4 examples/train_metric_multiview.py --config configs/multiview_metric.yaml --ddp
# DDP 评估（all-gather 汇总嵌入）
torchrun --nproc_per_node=4 examples/eval_retrieval_multiview.py --config configs/multiview_real.yaml \
  --ckpt runs/lstc_real_mv/best.pt --ddp
```

### 数据集准备
- 生成一个迷你数据集（支持多视角结构）：
```bash
uv run python examples/gen_toy_dataset.py --out ./toy_data --subjects 4 --seq-per-subject 3 --frames 20
```

- 目录建议：
```
/your_data_root/
  subject_0001/
    seq_0001/000001.png ...
    seq_0002/...
  subject_0002/...
```
- 建议：Resize 到 64x44、归一化到 [0,1]、采样/填充到固定 T 帧并保持时间顺序。

### 推荐超参
- 条带数 S：6–10（默认 8）
- 卷积核：kT=3，kH=7→5→3（逐层减小），kW=3
- LSTP top-k：2–4；优化器：AdamW(lr=3e-4, wd=0.05)、CosineLR、warmup=5
- 度量学习：PK 采样（如 P=8, K=4；DDP 下需整除）

### 常见问题 / FAQ
- 检测不到 CUDA（cuda? False）：
  - 误装了 CPU 版 PyTorch。请从 CUDA 轮子源（cu121/cu124）重装，并确认运行所用的 venv 与安装一致。
  - 显卡驱动过旧。更新主机/WSL 的 NVIDIA 驱动。环境内无需安装 CUDA Toolkit。
- CUDA 显存不足（OOM）：
  - 减小 `batch_size`、`seq_len`、`height/width`、或 `base_channels`；开启 `--amp`。
  - DDP 下的 batch-size 为“每卡”大小。
- 提示 P*K 需被 world_size 整除：
  - 调整 `batch_p` / `batch_k`，保证 P×K % world_size == 0。
- 数据为空 / 未找到：
  - 检查 `data_root` 目录布局与图像扩展名；必要时调整数据扫描的最小帧数。
- 设定随机种子后训练变慢：
  - 为了可复现，启用了确定性 cuDNN，会禁用部分启发式优化。可按需在 `set_seed` 放宽。
- DDP 启动卡住：
  - 确认用 `torchrun` 启动、端口畅通、各 rank 看到一致的文件；避免不同 rank 使用不同的 P×K。

### 参考
- GaitSet（AAAI’19）、GaitPart（CVPR’20）、GaitGL（CVIU’21）
- I3D（CVPR’17）、R(2+1)D（CVPR’18）、P3D（ICCV’17）
- 非对称/分解卷积（ACNet/Rep 系）

### 许可
MIT
