# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from qai_hub_models.scorecard.envvars import (
    EnabledModelsEnvvar,
    SpecialModelSetting,
    StaticModelsDirEnvvar,
)
from qai_hub_models.scorecard.internal_config_yaml import (
    LLMWeekendGroup,
    QAIHMModelInternalConfig,
    TestRunnerSplit,
    get_downloadable_llm_model_ids,
    get_week_model_ids,
)
from qai_hub_models.scorecard.static.model_config import DEFAULT_MODELS_DIR
from qai_hub_models.utils.path_helpers import MODEL_IDS as PYTORCH_RECIPE_MODEL_IDS
from qai_hub_models.utils.path_helpers import QAIHM_PACKAGE_ROOT


def get_static_bench_models_path() -> Path:
    return QAIHM_PACKAGE_ROOT / "scorecard" / "static" / "static_bench_models.txt"


def get_pytorch_bench_models_path() -> Path:
    return (
        QAIHM_PACKAGE_ROOT / "scorecard" / "static" / "pytorch_bench_models_float.txt"
    )


def get_pytorch_w8a8_bench_models_path() -> Path:
    return QAIHM_PACKAGE_ROOT / "scorecard" / "static" / "pytorch_bench_models_w8a8.txt"


def get_pytorch_w8a16_bench_models_path() -> Path:
    return (
        QAIHM_PACKAGE_ROOT / "scorecard" / "static" / "pytorch_bench_models_w8a16.txt"
    )


def get_all_static_models(
    models_path: Path = DEFAULT_MODELS_DIR, include_examples: bool = False
) -> list[str]:
    return [
        x[:-5]
        for x in os.listdir(models_path)
        if x.endswith(".yaml") and (include_examples or not x.endswith("_example.yaml"))
    ]


def get_bench_static_models() -> list[str]:
    with open(get_static_bench_models_path()) as f:
        return f.read().strip().split("\n")


@lru_cache
def get_bench_pytorch_w8a8_models() -> list[str]:
    with open(get_pytorch_w8a8_bench_models_path()) as f:
        return f.read().strip().split("\n")


@lru_cache
def get_bench_pytorch_w8a16_models() -> list[str]:
    with open(get_pytorch_w8a16_bench_models_path()) as f:
        return f.read().strip().split("\n")


def get_bench_pytorch_models() -> list[str]:
    with open(get_pytorch_bench_models_path()) as f:
        return f.read().strip().split("\n")


def get_all_bench_models() -> list[str]:
    return get_bench_static_models() + get_bench_pytorch_models()


@lru_cache
def get_pytorch_no_llm_model_ids() -> frozenset[str]:
    """Pytorch recipes with internal_config.test_split != llm."""
    return frozenset(
        m
        for m in PYTORCH_RECIPE_MODEL_IDS
        if QAIHMModelInternalConfig.from_model(m).test_split != TestRunnerSplit.LLM
    )


def validate_and_split_enabled_models(
    models: set[str | SpecialModelSetting] | None = None,
    static_models_dir: Path | None = None,
) -> tuple[set[str], set[str]]:
    """
    From the given models, extract enabled model IDs.

    Parameters
    ----------
    models
        Model IDs to enable. Can include special values (ALL, BENCH, PYTORCH).
        If None, retrieved from the current environment variable (QAIHM_TEST_MODELS).
    static_models_dir
        The location of static model definitions.
        If None, retrieved from the current environment variable (QAIHM_TEST_STATIC_MODELS_DIR).

    Returns
    -------
    torch_model_ids : set[str]
        Set of enabled torch model IDs.
    static_model_ids : set[str]
        Set of enabled static model IDs.

    Raises
    ------
    ValueError
        If a model id in `models` is unknown.
    """
    if models is None:
        models = EnabledModelsEnvvar.get()
    all_static_model_ids = set(
        get_all_static_models(
            static_models_dir
            if static_models_dir is not None
            else StaticModelsDirEnvvar.get()
        )
    )
    all_torch_model_ids = set(PYTORCH_RECIPE_MODEL_IDS)
    enabled_static_model_ids: set[str] = set()
    enabled_torch_model_ids: set[str] = set()
    for model_id in models:
        if model_id == SpecialModelSetting.ALL:
            enabled_static_model_ids = all_static_model_ids
            enabled_torch_model_ids = all_torch_model_ids
            break
        if model_id == SpecialModelSetting.STATIC:
            enabled_static_model_ids = all_static_model_ids
        elif model_id == SpecialModelSetting.BENCH:
            enabled_static_model_ids = enabled_static_model_ids.union(
                get_bench_static_models()
            )
            enabled_torch_model_ids = enabled_torch_model_ids.union(
                get_bench_pytorch_models()
            )
        elif model_id == SpecialModelSetting.PYTORCH:
            enabled_torch_model_ids = all_torch_model_ids
        elif model_id == SpecialModelSetting.PYTORCH_NO_LLM:
            enabled_torch_model_ids = enabled_torch_model_ids.union(
                get_pytorch_no_llm_model_ids()
            )
        elif model_id == SpecialModelSetting.LLM_WEEK1:
            enabled_torch_model_ids = enabled_torch_model_ids.union(
                get_week_model_ids(LLMWeekendGroup.WEEK1)
            )
        elif model_id == SpecialModelSetting.LLM_WEEK2:
            enabled_torch_model_ids = enabled_torch_model_ids.union(
                get_week_model_ids(LLMWeekendGroup.WEEK2)
            )
        elif model_id == SpecialModelSetting.LLM_NO_WEEK:
            enabled_torch_model_ids = enabled_torch_model_ids.union(
                get_week_model_ids(LLMWeekendGroup.NO_WEEK)
            )
        elif model_id == SpecialModelSetting.LLM_DOWNLOADABLE:
            enabled_torch_model_ids = enabled_torch_model_ids.union(
                get_downloadable_llm_model_ids()
            )
        else:
            model_id = model_id.lower()
            if model_id in all_static_model_ids:
                enabled_static_model_ids.add(model_id)
            elif model_id in all_torch_model_ids:
                enabled_torch_model_ids.add(model_id)
            else:
                raise ValueError(f"Unknown model_id: {model_id}")

    return enabled_torch_model_ids, enabled_static_model_ids
