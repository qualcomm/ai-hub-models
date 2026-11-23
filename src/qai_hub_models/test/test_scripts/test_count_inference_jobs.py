# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from qai_hub_models.scorecard.device import DEFAULT_SCORECARD_DEVICE
from qai_hub_models.scorecard.envvars import (
    EnabledDevicesEnvvar,
    EnabledPathsEnvvar,
    EnabledPrecisionsEnvvar,
    IgnoreKnownFailuresEnvvar,
    SpecialModelSetting,
)
from qai_hub_models.scripts.count_inference_jobs import count_inference_jobs
from qai_hub_models.utils.set_env import set_temp_env


def get_test_env(
    devices: str = "all",
    runtimes: str = "default",
    precisions: str = "default",
    run_all_skipped_jobs: bool = True,
) -> dict[str, str | None]:
    return {
        EnabledDevicesEnvvar.VARNAME: devices,
        EnabledPathsEnvvar.VARNAME: runtimes,
        EnabledPrecisionsEnvvar.VARNAME: precisions,
        IgnoreKnownFailuresEnvvar.VARNAME: IgnoreKnownFailuresEnvvar.serialize(
            run_all_skipped_jobs
        ),
    }


def test_count_inference_jobs_all_models() -> None:
    # Verify job collection works for all models
    with set_temp_env(get_test_env()):
        total_jobs_default, *_ = count_inference_jobs({SpecialModelSetting.ALL})
    with set_temp_env(get_test_env(run_all_skipped_jobs=False)):
        total_jobs_not_skipped, *_ = count_inference_jobs({SpecialModelSetting.ALL})
    assert total_jobs_default > total_jobs_not_skipped


def test_multiple_device_settings() -> None:
    with set_temp_env(get_test_env(devices="cs_x_elite,cs_6490")):
        count_inference_jobs({SpecialModelSetting.ALL})

    with set_temp_env(get_test_env(devices=f"{DEFAULT_SCORECARD_DEVICE.name},cs_6490")):
        count_inference_jobs({SpecialModelSetting.ALL})


def test_count_inference_jobs_filtering() -> None:
    with set_temp_env(get_test_env()):
        (
            total_jobs,
            jobs_by_device,
            job_by_device_form_factor,
            jobs_by_path,
            jobs_by_recipe,
            jobs_by_recipe_component,
            jobs_by_static_model,
        ) = count_inference_jobs({"resnet50"})
        assert len(jobs_by_device) > 1
        assert len(job_by_device_form_factor) > 1
        assert len(jobs_by_path) > 1
        assert len(jobs_by_recipe) == 1
        assert len(jobs_by_recipe_component) == 0
        assert len(jobs_by_static_model) == 0
    with set_temp_env(get_test_env(precisions="float")):
        total_jobs_fp_only, _, _, _, jobs_by_recipe, _, _ = count_inference_jobs(
            {"resnet50"}, show_precision_in_summary=True
        )
        assert len(jobs_by_recipe) == 1
    with set_temp_env(get_test_env(runtimes="tflite")):
        total_jobs_tflite, _, _, jobs_by_path, jobs_by_recipe, _, _ = (
            count_inference_jobs({"resnet50"}, show_precision_in_summary=True)
        )
        assert len(jobs_by_path) == 1
        assert (
            len(jobs_by_recipe) == 2
        )  # 1 recipe per supported precision, since show_precision_in_summary is True
    with set_temp_env(get_test_env(devices=DEFAULT_SCORECARD_DEVICE.name)):
        total_jobs_default_device, jobs_by_device, _, _, _, _, _ = count_inference_jobs(
            {"resnet50"}
        )
        assert len(jobs_by_device) == 1
    with set_temp_env(
        get_test_env(devices=DEFAULT_SCORECARD_DEVICE.name, runtimes="tflite")
    ):
        total_jobs_default_device_tflite, *_ = count_inference_jobs({"resnet50"})
    with set_temp_env(
        get_test_env(
            devices=DEFAULT_SCORECARD_DEVICE.name, runtimes="tflite", precisions="float"
        )
    ):
        total_jobs_default_device_tflite_fp, *_ = count_inference_jobs({"resnet50"})

    # Verify filtering by device / runtime / precision works
    assert total_jobs > total_jobs_fp_only
    assert total_jobs > total_jobs_tflite
    assert total_jobs > total_jobs_default_device
    assert total_jobs_fp_only > total_jobs_default_device
    assert total_jobs_tflite > total_jobs_default_device
    assert total_jobs_default_device > total_jobs_default_device_tflite
    assert total_jobs_default_device_tflite > total_jobs_default_device_tflite_fp


def test_count_inference_jobs_static_models() -> None:
    with set_temp_env(get_test_env()):
        total_jobs, *_ = count_inference_jobs({"cdcn_torchscript"})
    assert total_jobs > 0


def test_count_inference_jobs_components() -> None:
    with set_temp_env(get_test_env()):
        (
            _,
            _,
            _,
            _,
            jobs_by_recipe,
            jobs_by_recipe_component,
            _,
        ) = count_inference_jobs({"mediapipe_face"})
        assert len(jobs_by_recipe) == 1
        assert len(jobs_by_recipe_component) == 1
        assert len(jobs_by_recipe_component["mediapipe_face"]) == 2

        (
            _,
            _,
            _,
            _,
            jobs_by_recipe,
            jobs_by_recipe_component,
            _,
        ) = count_inference_jobs({"mediapipe_face"}, show_precision_in_summary=True)
        assert len(jobs_by_recipe) == 2
        assert len(jobs_by_recipe_component) == 2
        assert len(jobs_by_recipe_component["mediapipe_face (float)"]) == 2
        assert len(jobs_by_recipe_component["mediapipe_face (w8a8)"]) == 2
