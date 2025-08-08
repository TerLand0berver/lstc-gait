from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple
import random

from examples.dataset_silhouette import scan_silhouette_root, GaitSilhouetteDataset, SequenceRecord


@dataclass
class MultiViewIndex:
    view_id: int
    seq_index: int  # index within the per-view records list


class MultiViewGaitDataset:
    """
    A light wrapper that merges multiple silhouette roots (each a 'view') into a single dataset.
    It keeps track of (view_id, seq_index) and provides per-view datasets for actual tensor loading.
    """

    def __init__(self, roots: List[Path], seq_len: int = 30, height: int = 64, width: int = 44, seed: int = 42):
        self.roots = [Path(r) for r in roots]
        self.seq_len = seq_len
        self.height = height
        self.width = width
        self.random = random.Random(seed)

        self.view_records: List[List[SequenceRecord]] = []
        self.view_label_maps: List[Dict[str, int]] = []
        self.view_datasets: List[GaitSilhouetteDataset] = []
        self.indices: List[MultiViewIndex] = []
        self.global_label_map: Dict[str, int] = {}

        for v, root in enumerate(self.roots):
            records, label_map = scan_silhouette_root(root, min_frames=2)
            self.view_records.append(records)
            self.view_label_maps.append(label_map)
            self.view_datasets.append(GaitSilhouetteDataset(records, seq_len=seq_len, height=height, width=width, sampling="uniform"))
            for i in range(len(records)):
                self.indices.append(MultiViewIndex(view_id=v, seq_index=i))

        # Build global label map by subject folder name across views
        subjects = sorted({rec.label_name for recs in self.view_records for rec in recs})
        self.global_label_map = {name: i for i, name in enumerate(subjects)}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        mv = self.indices[idx]
        # Load tensor via per-view dataset, but compute a global label by name
        x, _ = self.view_datasets[mv.view_id][mv.seq_index]
        rec = self.view_records[mv.view_id][mv.seq_index]
        y_global = self.global_label_map[rec.label_name]
        return x, y_global, mv.view_id
