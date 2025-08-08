from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import re

from examples.dataset_silhouette import GaitSilhouetteDataset, SequenceRecord


@dataclass
class OUMVLPRecord(SequenceRecord):
    subject: str
    view: str
    path: Path


def _parse_ou_path(seq_dir: Path) -> Optional[tuple[str, str]]:
    # Support common structure: root/subject/view/frames
    parts = seq_dir.parts
    if len(parts) >= 2:
        view = parts[-2]
        subj = parts[-3] if len(parts) >= 3 else None
        if subj and re.match(r"^\d{4}$", subj) and re.match(r"^\d{3}$", view):
            return subj, view
    return None


def scan_ou_mvlp(root: Path, include_views: Optional[List[str]] = None, min_frames: int = 1) -> List[OUMVLPRecord]:
    root = Path(root)
    assert root.is_dir(), f"Not a directory: {root}"
    recs: List[OUMVLPRecord] = []
    for seq_dir in root.rglob("*"):
        if not seq_dir.is_dir():
            continue
        parsed = _parse_ou_path(seq_dir)
        if parsed is None:
            continue
        subj, view = parsed
        if include_views and view not in include_views:
            continue
        frames = sorted([p for p in seq_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png",".jpg",".jpeg",".bmp"}])
        if len(frames) < min_frames:
            continue
        label_id = int(subj)
        recs.append(OUMVLPRecord(label_id=label_id, label_name=subj, frames=frames, subject=subj, view=view, path=seq_dir))
    recs.sort(key=lambda r: (int(r.subject), r.view, r.path.as_posix()))
    return recs


def build_ou_dataset(records: List[OUMVLPRecord], seq_len: int, height: int, width: int) -> GaitSilhouetteDataset:
    generic = [SequenceRecord(label_id=r.label_id, label_name=r.label_name, frames=r.frames) for r in records]
    return GaitSilhouetteDataset(generic, seq_len=seq_len, height=height, width=width, sampling="uniform")
