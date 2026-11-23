# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.scorecard.envvars import SpecialModelSetting
from qai_hub_models.scorecard.static.limits import (
    BU_INFERENCE_JOB_LIMIT,
    BU_INFERENCE_JOB_LIMIT_BY_FORM_FACTOR,
)
from qai_hub_models.scorecard.static.model_config import ScorecardModelConfig
from qai_hub_models.scripts.count_inference_jobs import (
    count_inference_jobs,
    inference_job_counts_to_printstr,
)


def _get_failure_str(
    job_count_str: str, failure_msg: str, bu_owner: ScorecardModelConfig.BU
) -> str:
    return (
        f"\nSUMMARY OF STATIC MODEL JOBS OWNED BY BU '{bu_owner.value.upper()}'\n\n"
        f"{job_count_str}\n"
        "\n\n"
        "!!!!!!!!!!!!\n\n"
        f"FAILED: {failure_msg}\n\n"
        "See above for a detailed breakdown of inference jobs per model. The above can be replicated on your machine with this command:\n"
        f"    python qai_hub_models/scripts/count_inference_jobs.py --bu-owner {bu_owner.value}\n\n"
        "!!!!!!!!!!!!\n"
    )


def test_verify_bu_job_limits() -> None:
    for bu in ScorecardModelConfig.BU:
        if bu == ScorecardModelConfig.BU.AI_HUB:
            continue  # no limit for this BU

        (
            total_jobs,
            jobs_by_device,
            jobs_by_device_form_factor,
            jobs_by_path,
            _,
            _,
            jobs_by_static_model,
        ) = count_inference_jobs({SpecialModelSetting.ALL}, bu_owner=bu)
        printstr = inference_job_counts_to_printstr(
            total_jobs,
            jobs_by_device,
            jobs_by_device_form_factor,
            jobs_by_path,
            {},
            {},
            jobs_by_static_model,
        )
        assert total_jobs <= BU_INFERENCE_JOB_LIMIT, _get_failure_str(
            printstr,
            f"Total static model job count for BU '{bu.value.upper()}' ({total_jobs}) is greater than the per-BU inference job limit ({BU_INFERENCE_JOB_LIMIT})",
            bu,
        )

        for ff, count in jobs_by_device_form_factor.items():
            if limit := BU_INFERENCE_JOB_LIMIT_BY_FORM_FACTOR.get(ff):
                assert count <= limit, _get_failure_str(
                    printstr,
                    f"Static model job count for BU '{bu.value.upper()}' on device form factor '{ff.value}' ({count}) is greater than the {ff.value} form factor limit ({limit})",
                    bu,
                )
