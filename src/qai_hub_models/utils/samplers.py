# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import Sampler

__all__ = [
    "EveryNSampler",
    "every_n_stride",
    "grouped_shuffled_order",
    "grouped_strided_order",
    "shuffled_order",
    "strided_order",
]


class EveryNSampler(Sampler):
    """Samples every stride-th item deterministically from a torch dataset.

    Picks indices offset, offset + stride, offset + 2 * stride, ... so that
    `count` items are returned, spread evenly across the dataset.

    Parameters
    ----------
    stride
        Gap between consecutive picked indices.
    count
        Number of indices to pick.
    offset
        Index of the first pick.
    """

    def __init__(self, stride: int, count: int, offset: int = 0) -> None:
        self.stride = stride
        self.count = count
        self.offset = offset

    def __iter__(self) -> Iterator[int]:
        return iter(
            range(self.offset, self.offset + self.count * self.stride, self.stride)
        )

    def __len__(self) -> int:
        return self.count


def every_n_stride(dataset_size: int, count: int) -> int:
    """
    The stride for picking `count` evenly spread indices out of a dataset.

    Parameters
    ----------
    dataset_size
        Number of indices in the dataset.
    count
        Number of indices to pick.

    Returns
    -------
    stride : int
        `dataset_size // count`, rounded down, and at least 1.
    """
    return max(1, dataset_size // max(1, count))


def strided_order(
    dataset_size: int, batch_size: int, num_samples: int | None = None
) -> list[int]:
    """
    Order the indices of a dataset so each batch is an evenly spread sample of it.

    The stride is `dataset_size // batch_size`, rounded down to a whole number
    (at least 1). Batch k takes every stride-th index starting at index k, so
    batches are disjoint and each spans the whole dataset. Indices left over
    after all strided batches (fewer than one batch's worth) follow in
    ascending order. The order is cut into batches by the caller, `batch_size`
    indices at a time.

    Parameters
    ----------
    dataset_size
        Number of indices in the dataset; the order is a permutation of
        `range(dataset_size)`.
    batch_size
        Number of indices in each evenly spread batch. Also sets the stride, so a
        batch larger than half the dataset gets a stride of 1 and is just
        consecutive indices.
    num_samples
        If set, return only the first `num_samples` indices, and stop building
        batches once enough exist. If None, return every index.

    Returns
    -------
    order : list[int]
        Dataset indices. The first `num_samples` of them, or all of them.
    """
    stride = every_n_stride(dataset_size, batch_size)
    batch_len = min(batch_size, dataset_size)
    order: list[int] = []
    for offset in range(stride):
        if num_samples is not None and len(order) >= num_samples:
            return order[:num_samples]
        order.extend(
            i for i in EveryNSampler(stride, batch_len, offset) if i < dataset_size
        )
    if num_samples is not None and len(order) >= num_samples:
        return order[:num_samples]
    used = set(order)
    order.extend(i for i in range(dataset_size) if i not in used)
    return order


def grouped_strided_order(groups: list[list[int]], batch_size: int) -> list[int]:
    """
    Order a dataset's indices for batching when some samples must stay together.

    Some datasets have several samples that only make sense evaluated together,
    for example every person crop of one COCO image (the keypoint metric matches
    detections within an image) or every view of one Kinetics video (view scores
    are averaged per video). Evenly striding over raw sample indices would split
    those samples across different batches. This function strides over the
    groups instead, then lists each group's indices side by side.

    The result contains every index in `groups` exactly once. Cut it into
    consecutive batches of `batch_size` and each batch holds whole groups that
    are spread evenly across the whole dataset, not clustered at its start. The
    cut is by index count, so when group sizes vary a batch boundary can still
    land inside a group; the rest of that group then starts the next batch.

    Parameters
    ----------
    groups
        One list per group, holding the dataset indices that belong together
        (e.g. the indices of all crops of one image). Groups are strided in the
        order given and indices within a group keep their order.
    batch_size
        How many dataset indices one batch should hold. Used to decide how many
        whole groups make up a batch, from the average group size.

    Returns
    -------
    order : list[int]
        All dataset indices, grouped as described above.
    """
    if not groups:
        return []
    avg_group_size = sum(len(g) for g in groups) / len(groups)
    groups_per_batch = max(1, round(batch_size / avg_group_size))
    return [i for g in strided_order(len(groups), groups_per_batch) for i in groups[g]]


def grouped_shuffled_order(groups: list[list[int]], seed: int) -> list[int]:
    """All indices in `groups`, whole groups in a seeded random order."""
    return [i for g in shuffled_order(len(groups), seed) for i in groups[g]]


def shuffled_order(dataset_size: int, seed: int) -> list[int]:
    """Every index in `range(dataset_size)`, shuffled reproducibly with `seed`."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(dataset_size, generator=generator).tolist()
