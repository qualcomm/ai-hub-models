# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import os

import pytest

from qai_hub_models import Precision
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.scorecard.device import cs_8_gen_3
from qai_hub_models.scorecard.devices_and_chipsets_yaml import load_similar_devices
from qai_hub_models.scorecard.path_profile import ScorecardInferencePath
from qai_hub_models.scorecard.perf_yaml import (
    QAIHMModelPerf,
)
from qai_hub_models.scorecard.results.scorecard_summary import (
    ModelTestConfig,
)
from qai_hub_models.scorecard.results.yaml import (
    ComponentNamesYaml,
    GraphNamesYaml,
    InferenceScorecardJobYaml,
    get_model_component_and_graph_names,
)
from qai_hub_models.utils.hub_clients import deployment_is_prod

EXPECTED_MODEL_CARD = QAIHMModelPerf.from_yaml(
    os.path.join(os.path.dirname(__file__), "perf_gt.yaml"), "trocr"
)


def test_generate_perf(hub_test_deployment: str) -> None:
    if not deployment_is_prod(hub_test_deployment):
        pytest.skip(
            f"This test uses jobs only accessible on production AI Hub Workbench. Enabled deployment: {hub_test_deployment}"
        )

    # Verify the model card is made correctly given the info
    job_ids = InferenceScorecardJobYaml.from_file(
        os.path.join(os.path.dirname(__file__), "profile_job_ids.yaml")
    )

    # Create enabled paramaterizations for this test
    test_params = ModelTestConfig(
        model_id="trocr",
        component_names=["encoder", "decoder"],
        graph_names=None,
        component_graph_names=None,
        profile_component_names=["encoder", "decoder"],
        inference_tests=[
            (Precision.float, ScorecardInferencePath.TFLITE, cs_8_gen_3),
            (Precision.float, ScorecardInferencePath.ONNX, cs_8_gen_3),
        ],
        enabled_paths={},
    )

    # Get summaries for this model and its components.
    summaries = test_params.get_all_export_test_summaries(
        None,
        None,
        None,
        None,
        job_ids,
    )

    model_card = QAIHMModelPerf("trocr")
    for summary in summaries:
        summary.add_to_perf(model_card)

    model_card.sort_entries()
    EXPECTED_MODEL_CARD.sort_entries()
    assert model_card == EXPECTED_MODEL_CARD


def _build_perf_card(model_id: str) -> QAIHMModelPerf:
    """Build a perf card from the checked-in intermediate job IDs."""
    job_ids = InferenceScorecardJobYaml.from_intermediates()
    manifest = QAIHMModelManifest.from_model(model_id)

    component_names, graph_names, component_graph_names = (
        get_model_component_and_graph_names(
            model_id,
            ComponentNamesYaml.from_test_artifacts(),
            GraphNamesYaml.from_test_artifacts(),
        )
    )
    test_params = ModelTestConfig.from_recipe_model(
        manifest, component_names, graph_names, component_graph_names
    )
    summaries = test_params.get_all_export_test_summaries(
        None,
        None,
        None,
        None,
        job_ids,
    )

    model_card = QAIHMModelPerf(model_id)
    for summary in summaries:
        summary.add_to_perf(model_card, include_failures=False)
    model_card.apply_similar_devices(load_similar_devices())
    return model_card


@pytest.mark.skipif(
    not os.environ.get("RUN_SLOW_TESTS"),
    reason="Slow (~2 min, requires prod Hub). Set RUN_SLOW_TESTS=1 to enable.",
)
def test_generate_perf_with_similar_devices(hub_test_deployment: str) -> None:
    if not deployment_is_prod(hub_test_deployment):
        pytest.skip(
            f"This test uses jobs only accessible on production AI Hub Workbench. Enabled deployment: {hub_test_deployment}"
        )

    model_card = _build_perf_card("inception_v3")
    expected = QAIHMModelPerf.from_model("inception_v3")
    model_card.sort_entries()
    expected.sort_entries()
    assert str(model_card) == str(expected)
