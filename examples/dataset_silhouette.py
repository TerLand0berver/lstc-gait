from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class SequenceRecord:
    label_id: int
    label_name: str
    frames: List[Path]


def scan_silhouette_root(root: Path, min_frames: int = 1) -> Tuple[List[SequenceRecord], Dict[str, int]]:
    """
    Scan a directory organized as:
        root/
          subject_0001/
            seq_0001/
              000001.png ...
            seq_0002/
              ...
          subject_0002/
            ...
    Returns sequence records and a label_name->label_id mapping.
    """
    root = Path(root)
    assert root.is_dir(), f"Not a directory: {root}"

    subjects = sorted([p for p in root.iterdir() if p.is_dir()])
    label_map: Dict[str, int] = {s.name: i for i, s in enumerate(subjects)}

    records: List[SequenceRecord] = []
    for s in subjects:
        label_id = label_map[s.name]
        # Prefer subdirectories as sequences; fallback to frames directly under subject
        seq_dirs = [d for d in s.iterdir() if d.is_dir()]
        if not seq_dirs:
            frames = sorted([p for p in s.iterdir() if p.suffix.lower() in IMG_EXTS])
            if len(frames) >= min_frames:
                records.append(SequenceRecord(label_id=label_id, label_name=s.name, frames=frames))
            continue
        for d in sorted(seq_dirs):
            frames = sorted([p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])
            if len(frames) >= min_frames:
                records.append(SequenceRecord(label_id=label_id, label_name=s.name, frames=frames))
    return records, label_map


class GaitSilhouetteDataset(Dataset):
    def __init__(
        self,
        records: List[SequenceRecord],
        seq_len: int = 30,
        height: int = 64,
        width: int = 44,
        sampling: str = "uniform",  # "uniform" or "rand_window"
        grayscale: bool = True,
        pad_mode: str = "repeat_last",  # or "zero"
        seed: int = 42,
        return_index: bool = False,
    ) -> None:
        super().__init__()
        assert seq_len >= 1
        assert sampling in {"uniform", "rand_window"}
        assert pad_mode in {"repeat_last", "zero"}

        self.records = records
        self.seq_len = seq_len
        self.height = height
        self.width = width
        self.sampling = sampling
        self.grayscale = grayscale
        self.pad_mode = pad_mode
        self.rng = random.Random(seed)
        self.return_index = return_index

    def __len__(self) -> int:
        return len(self.records)

    def _choose_indices(self, num_frames: int) -> List[int]:
        L = self.seq_len
        if num_frames >= L:
            if self.sampling == "uniform":
                # inclusive linspace across [0, num_frames-1]
                idx = np.linspace(0, num_frames - 1, num=L, dtype=np.int64).tolist()
                return idx
            else:  # rand_window
                start = self.rng.randint(0, num_frames - L)
                return list(range(start, start + L))
        else:
            # need padding
            base = list(range(num_frames))
            if self.pad_mode == "repeat_last":
                pad = [num_frames - 1] * (L - num_frames)
            else:
                pad = [-1] * (L - num_frames)  # will map to zeros
            return base + pad

    def _load_frame(self, path: Optional[Path]) -> np.ndarray:
        if path is None:
            return np.zeros((self.height, self.width), dtype=np.float32)
        img = Image.open(path)
        if self.grayscale:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        img = img.resize((self.width, self.height), resample=Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr / 255.0
        else:
            arr = arr[:, :, 0] / 255.0  # take first channel if RGB
        return arr

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        num_frames = len(rec.frames)
        indices = self._choose_indices(num_frames)
        frames: List[np.ndarray] = []
        for i in indices:
            if i == -1:
                frames.append(self._load_frame(None))
            else:
                frames.append(self._load_frame(rec.frames[i]))
        # (T, H, W) -> (1, T, H, W)
        arr = np.stack(frames, axis=0)
        x = torch.from_numpy(arr).unsqueeze(0).float()
        y = rec.label_id
        if self.return_index:
            return x, y, idx
        return x, y
