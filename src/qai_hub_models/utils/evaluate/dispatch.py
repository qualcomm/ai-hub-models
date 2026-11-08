# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Select and bind the evaluate pipeline for a recipe folder."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from qai_hub_models.utils.base_collection_model import CollectionModel
from qai_hub_models.utils.export.context import resolve_model_cls


def select_evaluate_pipeline(source_dir: Path) -> Callable[..., Any]:
    """Return the pipeline ``evaluate_model`` for *source_dir* with the recipe pre-bound.

    The right ``evaluate_model`` is chosen from the recipe's ``Model``
    class: :class:`CollectionModel` subclasses go through the per-component
    pipeline, everything else through the single-graph pipeline.
    """
    model_cls = resolve_model_cls(source_dir)
    if issubclass(model_cls, CollectionModel):
        from .collection_pipeline import evaluate_model as evaluate_collection_model

        pipeline_fn: Callable[..., Any] = evaluate_collection_model
    else:
        from .pipeline import evaluate_model as evaluate_single_model

        pipeline_fn = evaluate_single_model

    bound = partial(pipeline_fn, source_dir=source_dir)
    sig = inspect.signature(pipeline_fn)
    bound.__signature__ = sig.replace(  # type: ignore[attr-defined]
        parameters=[p for n, p in sig.parameters.items() if n != "source_dir"]
    )
    bound.__doc__ = pipeline_fn.__doc__
    return bound
