# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
What the scorecard stages need to export an LLM without the pytest flow: the
default checkpoint of the precision and the single-instantiation restriction.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from typing import Any
from unittest.mock import patch

from qai_hub_models import Precision
from qai_hub_models.models.templates.llm.checkpoint_stub import (
    llm_checkpoint_name,
    stub_llm_checkpoint_resolution,
)
from qai_hub_models.scorecard.utils.testing_export_eval import (
    _maybe_restrict_to_single_instantiation,
)
from qai_hub_models.utils.export.context import resolve_model_cls
from qai_hub_models.utils.path_helpers import QAIHM_MODELS_ROOT


def llm_export_kwargs(precision: Precision) -> dict[str, Any]:
    """Extra export_model kwargs that load the LLM's default checkpoint."""
    return {"checkpoint": llm_checkpoint_name(precision)}


def _restricted_from_pretrained(model_cls: type) -> Callable[..., Any]:
    original = model_cls.from_pretrained  # type: ignore[attr-defined]

    def from_pretrained(*args: Any, **kwargs: Any) -> Any:
        model = original(*args, **kwargs)
        _maybe_restrict_to_single_instantiation(model)
        return model

    return from_pretrained


@contextmanager
def llm_export_context(model_id: str, stub_checkpoint: bool = False) -> Iterator[None]:
    """
    Patch the LLM class for an export with the single-instantiation
    restriction. With `stub_checkpoint`, also skip the FP HuggingFace load (for
    exports that only re-derive names from existing jobs).
    """
    model_cls = resolve_model_cls(QAIHM_MODELS_ROOT / model_id)
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                model_cls, "from_pretrained", _restricted_from_pretrained(model_cls)
            )
        )
        if stub_checkpoint:
            stack.enter_context(stub_llm_checkpoint_resolution(model_cls))
        yield
