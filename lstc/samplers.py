from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable, List, Dict

from torch.utils.data import Sampler
import torch.distributed as dist


class PKSampler(Sampler[list[int]]):
    """
    Identity-aware PK sampler for classification+metric learning:
    - At each batch, sample P distinct identities and K sequences per identity (total P*K items).
    - Works with datasets where __getitem__ returns (tensor, label_id).
    """

    def __init__(self, labels: Iterable[int], batch_p: int, batch_k: int, seed: int = 42):
        self.labels = list(labels)
        self.batch_p = batch_p
        self.batch_k = batch_k
        self.random = random.Random(seed)

        # Build id -> indices map
        self.id_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, lab in enumerate(self.labels):
            self.id_to_indices[int(lab)].append(idx)
        self.ids = list(self.id_to_indices.keys())

    def __iter__(self):
        # Shuffle per identity indices
        for id_, lst in self.id_to_indices.items():
            self.random.shuffle(lst)

        id_pos = {id_: 0 for id_ in self.ids}
        ids = self.ids[:]
        self.random.shuffle(ids)

        batch: List[int] = []
        while True:
            if len(batch) == 0:
                # Draw P ids; if not enough remaining, reshuffle
                if len(ids) < self.batch_p:
                    ids = self.ids[:]
                    self.random.shuffle(ids)
                active_ids = [ids.pop() for _ in range(self.batch_p)]

            for id_ in active_ids:
                start = id_pos[id_]
                end = start + self.batch_k
                # If not enough samples left for this identity, reshuffle its list
                if end > len(self.id_to_indices[id_]):
                    self.random.shuffle(self.id_to_indices[id_])
                    start = 0
                    end = self.batch_k
                chosen = self.id_to_indices[id_][start:end]
                id_pos[id_] = end
                batch.extend(chosen)

            if len(batch) == self.batch_p * self.batch_k:
                # DDP-aware sharding of the batch (optional)
                if dist.is_available() and dist.is_initialized():
                    world_size = dist.get_world_size()
                    rank = dist.get_rank()
                    assert (self.batch_p * self.batch_k) % world_size == 0, "P*K must be divisible by world_size in DDP"
                    chunk = (self.batch_p * self.batch_k) // world_size
                    start = rank * chunk
                    end = start + chunk
                    yield batch[start:end]
                else:
                    yield batch
                batch = []

            # Stop condition: when most ids have exhausted a full pass
            if all(id_pos[i] >= len(self.id_to_indices[i]) for i in self.ids):
                break

    def __len__(self):
        # Approximate number of batches per epoch
        total = len(self.labels)
        num_batches = total // (self.batch_p * self.batch_k)
        # In DDP, each rank sees num_batches batches too (already sharded per-batch)
        return num_batches


class MultiViewPKSampler(Sampler[list[int]]):
    """
    PK sampler aware of multi-view datasets.
    - labels: list of identity labels for each index in the merged multi-view dataset
    - view_ids: list of view id (int) for each index; same length as labels
    - batch_p: number of distinct identities per batch
    - batch_k: number of samples per identity per batch
    - views_per_id: try to cover up to this many distinct views per identity in a batch (best-effort)
    """

    def __init__(self, labels: Iterable[int], view_ids: Iterable[int], batch_p: int, batch_k: int, views_per_id: int = 2, seed: int = 42, balance_across_views: bool = True):
        self.labels = list(map(int, labels))
        self.view_ids = list(map(int, view_ids))
        assert len(self.labels) == len(self.view_ids)
        self.batch_p = batch_p
        self.batch_k = batch_k
        self.views_per_id = max(1, views_per_id)
        self.random = random.Random(seed)
        self.balance_across_views = balance_across_views

        # Build id -> view -> indices map
        self.id_to_view_to_indices: Dict[int, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
        for idx, (lab, vid) in enumerate(zip(self.labels, self.view_ids)):
            self.id_to_view_to_indices[lab][vid].append(idx)
        self.ids = list(self.id_to_view_to_indices.keys())
        self.all_views = sorted({vid for m in self.id_to_view_to_indices.values() for vid in m.keys()})

    def __iter__(self):
        # Shuffle indices per id/view
        for id_, view_map in self.id_to_view_to_indices.items():
            for vid, lst in view_map.items():
                self.random.shuffle(lst)

        # Maintain cursors per id/view
        cursors: Dict[int, Dict[int, int]] = defaultdict(dict)
        for id_, view_map in self.id_to_view_to_indices.items():
            for vid in view_map.keys():
                cursors[id_][vid] = 0

        ids = self.ids[:]
        self.random.shuffle(ids)

        batch: List[int] = []
        view_counts: Dict[int, int] = {v: 0 for v in self.all_views}
        total_needed = self.batch_p * self.batch_k
        # Compute per-view target counts for balance (best effort)
        base = total_needed // max(1, len(self.all_views))
        rem = total_needed - base * max(1, len(self.all_views))
        target: Dict[int, int] = {v: base + (1 if i < rem else 0) for i, v in enumerate(self.all_views)}
        while True:
            if len(ids) < self.batch_p:
                ids = self.ids[:]
                self.random.shuffle(ids)
            active_ids = [ids.pop() for _ in range(self.batch_p)]

            for id_ in active_ids:
                view_map = self.id_to_view_to_indices[id_]
                available_views = list(view_map.keys())
                # Prefer views with largest deficit relative to target
                if self.balance_across_views:
                    available_views.sort(key=lambda v: (view_counts.get(v, 0) - target.get(v, 0)))
                else:
                    self.random.shuffle(available_views)
                chosen_views = available_views[: self.views_per_id]
                remaining = self.batch_k
                alloc: Dict[int, int] = {v: 0 for v in chosen_views}
                while remaining > 0 and chosen_views:
                    # pick the currently most under-filled view in batch so far
                    if self.balance_across_views:
                        chosen_views.sort(key=lambda v: (view_counts.get(v, 0) - target.get(v, 0)))
                    for v in chosen_views:
                        if remaining == 0:
                            break
                        alloc[v] += 1
                        remaining -= 1

                # Collect indices according to allocation
                for v, cnt in alloc.items():
                    if cnt <= 0:
                        continue
                    start = cursors[id_][v]
                    end = start + cnt
                    if end > len(view_map[v]):
                        # reshuffle this view list and wrap
                        self.random.shuffle(view_map[v])
                        start = 0
                        end = cnt
                    chosen = view_map[v][start:end]
                    cursors[id_][v] = end
                    batch.extend(chosen)
                    view_counts[v] = view_counts.get(v, 0) + len(chosen)

            if len(batch) == self.batch_p * self.batch_k:
                # DDP-aware sharding by slicing the batch
                if dist.is_available() and dist.is_initialized():
                    world_size = dist.get_world_size()
                    rank = dist.get_rank()
                    assert (self.batch_p * self.batch_k) % world_size == 0, "P*K must be divisible by world_size in DDP"
                    chunk = (self.batch_p * self.batch_k) // world_size
                    start = rank * chunk
                    end = start + chunk
                    yield batch[start:end]
                else:
                    yield batch
                batch = []
                view_counts = {v: 0 for v in self.all_views}

            # Stop condition: when all cursors reached ends (approximate)
            if all(
                all(cursors[id_][v] >= len(self.id_to_view_to_indices[id_][v]) for v in self.id_to_view_to_indices[id_])
                for id_ in self.ids
            ):
                break

    def __len__(self):
        total = len(self.labels)
        num_batches = total // (self.batch_p * self.batch_k)
        return num_batches
