# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

from transformers import AutoConfig, AutoTokenizer

from qai_hub_models import Precision
from qai_hub_models.models.templates.llama3.model import (
    LlamaDynamicQuantizablePreSplitMixin,
)
from qai_hub_models.models.templates.llm.model import DynamicQuantizablePreSplitMixin
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.envvars import CheckpointEnvvar


def llm_checkpoint_name(precision: Precision) -> str:
    """The `checkpoint` argument that loads the default checkpoint of `precision`."""
    return f"DEFAULT_{str(precision).upper()}"


def _ensure_tokenizer_and_config(cls: Any, ckpt: Path) -> None:
    if not (ckpt / "tokenizer.json").exists():
        AutoTokenizer.from_pretrained(cls.FPModel.hf_repo_name).save_pretrained(ckpt)
    if not (ckpt / "config.json").exists():
        AutoConfig.from_pretrained(cls.FPModel.hf_repo_name).save_pretrained(ckpt)


def _stub_resolve_zip_checkpoint(
    cls: Any, precision: Precision, host_device: object, fp_model: object
) -> tuple[str, None]:
    # Qwen3 (base DynamicQuantizablePreSplitMixin) publishes the full .zip
    # archive (dynamic ONNX + encodings + weights + tokenizer + config), so
    # reuse the real fetch_default_checkpoint rather than fetching a bare
    # model.encodings object that was never uploaded. Only the FP
    # HuggingFace load (skipped by not passing fp_model) needs stubbing.
    return cls.fetch_default_checkpoint(precision), None


def _stub_resolve_encodings_checkpoint(
    cls: Any, precision: Precision, host_device: object, fp_model: object
) -> tuple[str, None]:
    # Llama fetches encodings only and re-exports ONNX from the FP torch
    # model, so its asset store publishes a bare model.encodings.
    if CheckpointEnvvar.get():
        ckpt = Path(cls.fetch_default_checkpoint(precision))
    else:
        precision_checkpoint = cls.default_checkpoint[precision]
        ckpt = Path(
            CachedWebModelAsset.from_asset_store(
                cls.model_id,
                cls.model_asset_version,
                f"{precision_checkpoint}/model.encodings",
            ).fetch()
        ).parent
    _ensure_tokenizer_and_config(cls, ckpt)
    return str(ckpt), None


@contextmanager
def stub_llm_checkpoint_resolution(model_cls: type) -> Iterator[None]:
    """Patch resolve_default_checkpoint (base + Llama override) plus
    get_component_graph_input_spec / get_component_graph_hub_compile_options on
    model_cls so exporting an LLM from already-submitted jobs skips the FP
    HuggingFace load.
    """
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                DynamicQuantizablePreSplitMixin,
                "resolve_default_checkpoint",
                classmethod(_stub_resolve_zip_checkpoint),
            )
        )
        stack.enter_context(
            patch.object(
                LlamaDynamicQuantizablePreSplitMixin,
                "resolve_default_checkpoint",
                classmethod(_stub_resolve_encodings_checkpoint),
            )
        )
        stack.enter_context(
            patch.object(
                model_cls,
                "get_component_graph_input_spec",
                lambda self, component_name, graph_name, *a, **kw: {},
            )
        )
        stack.enter_context(
            patch.object(
                model_cls,
                "get_component_graph_hub_compile_options",
                lambda self, *a, **kw: "",
            )
        )
        yield
