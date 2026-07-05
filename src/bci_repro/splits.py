from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SplitItem:
    pair_id: str
    concept: str
    subject: str
    method: str


def split_indices(
    examples: Iterable[SplitItem],
    split_by: str,
    seed: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> dict[str, list[int]]:
    examples = list(examples)
    group_to_indices: dict[str, list[int]] = {}
    for idx, example in enumerate(examples):
        group = getattr(example, split_by)
        group_to_indices.setdefault(group, []).append(idx)

    rng = random.Random(seed)
    groups = list(group_to_indices)
    rng.shuffle(groups)
    train_cut = int(round(len(groups) * train_frac))
    val_cut = int(round(len(groups) * (train_frac + val_frac)))
    split_groups = {
        "train": set(groups[:train_cut]),
        "val": set(groups[train_cut:val_cut]),
        "test": set(groups[val_cut:]),
    }
    splits = {
        split: sorted(idx for group in groups_for_split for idx in group_to_indices[group])
        for split, groups_for_split in split_groups.items()
    }
    if not splits["val"] or not splits["test"]:
        raise ValueError(f"Split by {split_by!r} produced empty val/test split.")
    return splits

