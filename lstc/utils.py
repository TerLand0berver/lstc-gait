from __future__ import annotations

import os
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from torch.utils.tensorboard import SummaryWriter


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_main_process() -> bool:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


class CSVLogger:
    def __init__(self, path: Path, fieldnames: list[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        self._init_file()

    def _init_file(self):
        new_file = not self.path.exists()
        self.f = open(self.path, "a", newline="")
        self.writer = csv.DictWriter(self.f, fieldnames=self.fieldnames)
        if new_file:
            self.writer.writeheader()

    def log(self, row: Dict[str, Any]):
        self.writer.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()


@dataclass
class Checkpoint:
    model: dict
    classifier: Optional[dict] = None
    epoch: int = 0
    best: Optional[dict] = None
    num_classes: Optional[int] = None
    args: Optional[dict] = None


def save_checkpoint(state: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def try_load_checkpoint(path: Optional[str | Path]) -> Optional[dict]:
    if not path:
        return None
    p = Path(path)
    if p.is_file():
        return torch.load(p, map_location="cpu")
    return None


def create_tb_writer(log_dir: Path, enabled: bool) -> Optional[SummaryWriter]:
    if not enabled:
        return None
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


class ModelEma:
    """Exponential Moving Average (EMA) of model parameters.
    - Keeps a shadow copy of parameters updated as: ema = decay*ema + (1-decay)*param
    - Buffers are copied (no EMA) by default.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999, device: Optional[torch.device] = None):
        self.decay = decay
        self.module = self._clone_model(model)
        if device is not None:
            self.module.to(device)
        self.module.eval()

    @torch.no_grad()
    def _clone_model(self, model: torch.nn.Module) -> torch.nn.Module:
        clone = type(model)(**getattr(model, 'init_kwargs', {})) if hasattr(model, 'init_kwargs') else None
        if clone is None:
            # Fallback to deepcopy
            import copy
            clone = copy.deepcopy(model)
        clone.load_state_dict(model.state_dict(), strict=True)
        for p in clone.parameters():
            p.requires_grad_(False)
        return clone

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for k, v in ema_state.items():
            if k in model_state and v.dtype.is_floating_point:
                ema_state[k].mul_(self.decay).add_(model_state[k], alpha=1.0 - self.decay)
            elif k in model_state:
                ema_state[k].copy_(model_state[k])

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module):
        model.load_state_dict(self.module.state_dict(), strict=False)
