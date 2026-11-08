# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from pathlib import Path
from unittest import mock

import pytest

pytest.importorskip(
    "qualcomm_device_cloud_sdk",
    reason="This test requires the qualcomm_device_cloud_sdk package.",
)

from qai_hub_models.utils.devicefarm.backends.qdc import qdc


@pytest.mark.parametrize(
    ("job_timeout", "expected"),
    [(None, qdc.QDC_DEFAULT_JOB_RUN_TIMEOUT), (0, 0), (3600, 3600)],
)
def test_qdc_submit_bundle_job_timeout_defaults_only_when_none(
    job_timeout: int | None, expected: int, tmp_path: Path
) -> None:
    farm = qdc.QDCDeviceFarm.__new__(qdc.QDCDeviceFarm)
    entry = tmp_path / "a.txt"
    entry.write_text("x")
    with (
        mock.patch.object(qdc, "QDCDevice"),
        mock.patch.object(farm, "_upload_file", return_value="artifact"),
        mock.patch.object(farm, "_submit_automated_job", return_value="1") as submit,
    ):
        farm.submit_bundle(
            "Snapdragon X2 Elite CRD",
            [(str(entry), "a.txt")],
            "script",
            "job",
            job_timeout=job_timeout,
        )
    assert submit.call_args.kwargs["job_timeout"] == expected
