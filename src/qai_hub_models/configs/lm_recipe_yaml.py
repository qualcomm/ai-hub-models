# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import Field, model_serializer, model_validator

from qai_hub_models import Precision

# configs normally does not import from models, but lm_schema is a self-contained,
# dependency-light (pydantic-only) island -- not model code -- so the recipe can
# validate against it without a real layer inversion.
from qai_hub_models.models.templates.lm_schema import PrecisionSchema, Recipe
from qai_hub_models.utils.base_config import BaseQAIHMConfig
from qai_hub_models.utils.path_helpers import QAIHM_MODELS_ROOT

__all__ = ["LMQuantizationDetails", "LMRecipe"]

LM_RECIPE_FILENAME = "lm_recipe.yaml"


class LMQuantizationDetails(BaseQAIHMConfig):
    """The ``precision:`` + ``recipe:`` pair that produced one precision's checkpoint.

    Both halves are validated by the shared GenAI Lab schema (``lm_schema``):
    ``precision`` is the bit config (defaults reproduce AIHM's W4A16 contract), and
    ``recipe`` is the ordered technique chain (SeqMSE / AdaScale / Calibration / ...).
    Copied verbatim from the GenAI Lab config that produced the checkpoint.
    """

    # No default: a field that merely equals its default is stripped by
    # exclude_defaults, and an omitted `precision:` means the W4A16 contract, not
    # "nothing". _fill_precision supplies it so existing recipes still validate.
    precision: PrecisionSchema
    recipe: Recipe

    @model_validator(mode="before")
    @classmethod
    def _fill_precision(cls, data: Any) -> Any:
        if isinstance(data, dict) and "precision" not in data:
            return {**data, "precision": {}}
        return data


class LMRecipe(BaseQAIHMConfig):
    """Schema for a model's ``lm_recipe.yaml``.

    Portable GenAI Lab quant recipes keyed by Hub precision name (a subset of the
    model's ``supported_precisions``). LLM/VLM folders only. On disk this is a bare
    ``{precision: {precision:, recipe:}}`` mapping.
    """

    details: dict[Precision, LMQuantizationDetails] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _parse_without_details(cls, v: Any) -> Any:
        if isinstance(v, dict) and "details" not in v:
            return {"details": v}
        return v

    @model_serializer(mode="wrap")
    def _serialize_without_details(self, handler: Callable[[LMRecipe], dict]) -> Any:
        result = handler(self)
        return result.get("details", result)

    def __getitem__(self, precision: Precision) -> LMQuantizationDetails:
        return self.details[precision]

    def get(self, precision: Precision) -> LMQuantizationDetails | None:
        return self.details.get(precision)

    @staticmethod
    def get_yaml_path(model_id: str) -> Path:
        return QAIHM_MODELS_ROOT / model_id / LM_RECIPE_FILENAME

    @classmethod
    def from_model(cls, model_id: str) -> LMRecipe:
        """Load lm_recipe.yaml for the model; empty if the model has none."""
        path = cls.get_yaml_path(model_id)
        if not os.path.exists(path):
            return cls()
        return cls.from_yaml(path)

    def to_model_yaml(self, model_id: str) -> Path:
        path = self.get_yaml_path(model_id)
        self.to_yaml(path)
        return path
