from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re

from examples.dataset_silhouette import GaitSilhouetteDataset, SequenceRecord


@dataclass
class CasiaBRecord(SequenceRecord):
    subject: str
    condition: str  # nm/bg/cl
    cond_id: str    # 01..06 etc
    view: str       # 000..180
    path: Path      # sequence directory


def _parse_casia_b_path(seq_dir: Path) -> Optional[Tuple[str, str, str, str]]:
    # Support common layouts:
    # 1) root/subject/condition-id/view/frames
    # 2) root/subject/view/condition-id/frames
    parts = seq_dir.parts
    # Try pattern 1
    if len(parts) >= 4:
        cond = parts[-2] if re.match(r"^(nm|bg|cl)-\d{2}$", parts[-2]) else None
        view = parts[-3] if re.match(r"^\d{3}$", parts[-3]) else None
        subj = parts[-4]
        if cond and view:
            condition, cond_id = cond.split("-")
            return subj, condition, cond_id, view
    # Try pattern 2
    if len(parts) >= 4:
        cond = parts[-3] if re.match(r"^(nm|bg|cl)-\d{2}$", parts[-3]) else None
        view = parts[-2] if re.match(r"^\d{3}$", parts[-2]) else None
        subj = parts[-4]
        if cond and view:
            condition, cond_id = cond.split("-")
            return subj, condition, cond_id, view
    return None


def scan_casia_b(root: Path, include_views: Optional[List[str]] = None, include_conditions: Optional[List[str]] = None, min_frames: int = 1) -> List[CasiaBRecord]:
    root = Path(root)
    assert root.is_dir(), f"Not a directory: {root}"
    recs: List[CasiaBRecord] = []
    # Find all sequence directories that contain image frames
    for seq_dir in root.rglob("*"):
        if not seq_dir.is_dir():
            continue
        parsed = _parse_casia_b_path(seq_dir)
        if parsed is None:
            continue
        subj, condition, cond_id, view = parsed
        if include_views and view not in include_views:
            continue
        if include_conditions and condition not in include_conditions:
            continue
        frames = sorted([p for p in seq_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])
        if len(frames) < min_frames:
            continue
        label_name = subj
        label_id = int(subj)
        recs.append(CasiaBRecord(label_id=label_id, label_name=label_name, frames=frames, subject=subj, condition=condition, cond_id=cond_id, view=view, path=seq_dir))
    # Sort for determinism
    recs.sort(key=lambda r: (int(r.subject), r.condition, r.cond_id, r.view, r.path.as_posix()))
    return recs


def build_casia_b_dataset(records: List[CasiaBRecord], seq_len: int, height: int, width: int) -> GaitSilhouetteDataset:
    # Reuse generic dataset by mapping
    generic_recs: List[SequenceRecord] = [SequenceRecord(label_id=r.label_id, label_name=r.label_name, frames=r.frames) for r in records]
    return GaitSilhouetteDataset(generic_recs, seq_len=seq_len, height=height, width=width, sampling="uniform")
