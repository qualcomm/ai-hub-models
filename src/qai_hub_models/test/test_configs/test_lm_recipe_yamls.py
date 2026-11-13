# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import pytest

from qai_hub_models import Precision
from qai_hub_models.configs.lm_recipe_yaml import LMQuantizationDetails, LMRecipe
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.utils.path_helpers import MODEL_IDS

_MODELS_WITH_RECIPE = [m for m in MODEL_IDS if LMRecipe.get_yaml_path(m).exists()]


@pytest.mark.parametrize("model_id", _MODELS_WITH_RECIPE)
def test_recipe_is_consistent_with_manifest(model_id: str) -> None:
    """Recipes only exist for LLM/VLMs, and only for precisions the model supports."""
    recipe = LMRecipe.from_model(model_id)
    manifest = QAIHMModelManifest.from_model(model_id)
    assert recipe.details, f"{model_id}: empty lm_recipe.yaml should be deleted"
    assert manifest.model_type_llm, (
        f"{model_id}: lm_recipe.yaml requires model_type_llm: true in manifest.yaml"
    )
    unknown = set(recipe.details) - set(manifest.supported_precisions)
    assert not unknown, (
        f"{model_id}: lm_recipe.yaml has recipes for {sorted(str(p) for p in unknown)} "
        f"that are not in supported_precisions "
        f"({sorted(str(p) for p in manifest.supported_precisions)})"
    )


def _lm_details() -> LMQuantizationDetails:
    """A minimal valid recipe (contract W4A16 precision + a single Calibration)."""
    # model_validate, not the constructor: an omitted `precision` is filled by
    # _fill_precision, which is the path recipe files take.
    return LMQuantizationDetails.model_validate({"recipe": [{"name": "Calibration"}]})


class TestLMQuantizationDetails:
    def test_omitted_precision_is_re_emitted_in_full(self) -> None:
        # exclude_defaults deletes any field that merely equals its default -- and in
        # lm_schema the defaults *are* the W4A16 contract. The schema fills the
        # contract at validation instead, so a section that omits `precision:`
        # re-emits it fully fleshed out.
        details = _lm_details()
        dumped = details.model_dump(exclude_defaults=True)

        assert dumped["precision"]["activations"] == "int16"
        assert dumped["precision"]["kv_cache"] == "int8"
        assert dumped["precision"]["embedding"] == "int16"
        assert dumped["precision"]["lm_head"] == {"qtype": "int8", "granularity": "PCQ"}
        # Sugar forms the lm_schema before-validators widen are re-narrowed.
        assert dumped["precision"]["blocks"] == {"qtype": "int4", "granularity": "PCQ"}
        assert dumped["recipe"] == [{"name": "Calibration"}]
        assert LMQuantizationDetails.model_validate(dumped) == details

    def test_partial_precision_is_re_emitted_in_full(self) -> None:
        # A partially written block keeps what was authored and gains the rest.
        details = LMQuantizationDetails.model_validate(
            {
                "precision": {"lm_head": {"qtype": "int16"}},
                "recipe": [{"name": "Calibration"}],
            }
        )
        dumped = details.model_dump(exclude_defaults=True)

        assert dumped["precision"]["lm_head"] == {
            "qtype": "int16",
            "granularity": "PCQ",
        }
        assert dumped["precision"]["kv_cache"] == "int8"

    def test_serialization_keeps_components_for_vlms(self) -> None:
        # A recipe with a visual chain has to keep the component form; only the
        # single-component case collapses to the bare list.
        details = LMQuantizationDetails.model_validate(
            {
                "recipe": {
                    "backbone": [{"name": "Calibration"}],
                    "visual": [{"name": "Calibration"}],
                }
            }
        )
        dumped = details.model_dump(exclude_defaults=True)

        assert dumped["recipe"]["backbone"] == [{"name": "Calibration"}]
        assert dumped["recipe"]["visual"] == [{"name": "Calibration"}]
        assert LMQuantizationDetails.model_validate(dumped) == details


class TestLMRecipe:
    def test_missing_file_is_empty(self) -> None:
        assert LMRecipe.from_model("resnet50").details == {}

    def test_yaml_round_trip_is_a_bare_precision_mapping(self, tmp_path: Path) -> None:
        recipe = LMRecipe(details={Precision.w4a16: _lm_details()})
        path = tmp_path / "lm_recipe.yaml"
        recipe.to_yaml(path)

        assert path.read_text().startswith("w4a16:\n")
        assert LMRecipe.from_yaml(path) == recipe
        assert recipe.get(Precision.w4a16) == _lm_details()
        assert recipe.get(Precision.w4) is None
