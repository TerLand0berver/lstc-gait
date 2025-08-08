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
