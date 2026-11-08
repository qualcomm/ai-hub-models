# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import datetime
from pathlib import Path
from unittest import mock

import boto3
import pytest
from botocore.awsrequest import AWSPreparedRequest, AWSResponse, HTTPHeaders
from botocore.credentials import DeferredRefreshableCredentials
from qai_hub_models_cli._internal import aws as cli_aws

from qai_hub_models.utils.devicefarm.backends.aws import aws
from qai_hub_models.utils.devicefarm.backends.aws.aws import (
    AwsDeviceFarmConfig,
    _make_session,
)

OIDC_ENV = {
    "ACTIONS_ID_TOKEN_REQUEST_URL": "https://token.example/?x=1",
    "ACTIONS_ID_TOKEN_REQUEST_TOKEN": "gh-token",
}
RUN_ARN = "arn:aws:devicefarm:us-west-2:000000000000:run:project/run"


def _creds(expiry: datetime.datetime, key: str) -> dict[str, str]:
    return {
        "access_key": key,
        "secret_key": "secret",
        "token": "session",
        "expiry_time": expiry.isoformat(),
    }


def test_make_session_without_oidc_uses_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in OIDC_ENV:
        monkeypatch.delenv(var, raising=False)
    config = AwsDeviceFarmConfig(project_arn="p", role_arn="arn:role")
    with mock.patch.object(boto3, "Session") as session_cls:
        _make_session(config)
    session_cls.assert_called_once_with(profile_name=None, region_name="us-west-2")


def test_make_session_with_oidc_refreshes_expired_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var, value in OIDC_ENV.items():
        monkeypatch.setenv(var, value)
    now = datetime.datetime.now(datetime.timezone.utc)
    fetcher = mock.MagicMock()
    fetcher.return_value.fetch_credentials.side_effect = [
        _creds(now + datetime.timedelta(minutes=1), "first"),
        _creds(now + datetime.timedelta(hours=1), "second"),
    ]
    config = AwsDeviceFarmConfig(project_arn="p", role_arn="arn:role")
    with mock.patch.object(
        cli_aws, "AssumeRoleWithWebIdentityCredentialFetcher", fetcher
    ):
        session = _make_session(config)

    creds = session.get_credentials()
    assert isinstance(creds, DeferredRefreshableCredentials)
    assert fetcher.call_args.kwargs["role_arn"] == "arn:role"
    assert creds.get_frozen_credentials().access_key == "first"
    # The first credentials are inside the refresh window, so the next use re-assumes.
    assert creds.get_frozen_credentials().access_key == "second"


def test_long_lived_client_signs_each_call_with_refreshed_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var, value in OIDC_ENV.items():
        monkeypatch.setenv(var, value)
    now = datetime.datetime.now(datetime.timezone.utc)
    fetcher = mock.MagicMock()
    fetcher.return_value.fetch_credentials.side_effect = [
        _creds(now + datetime.timedelta(minutes=1), "AKIAFIRST"),
        _creds(now + datetime.timedelta(hours=1), "AKIASECOND"),
    ]
    config = AwsDeviceFarmConfig(project_arn="p", role_arn="arn:role")
    with mock.patch.object(
        cli_aws, "AssumeRoleWithWebIdentityCredentialFetcher", fetcher
    ):
        client = _make_session(config).client("devicefarm", region_name="us-west-2")

    signed_keys: list[str] = []

    def _capture(request: AWSPreparedRequest, **kwargs: object) -> AWSResponse:
        auth = request.headers["Authorization"]
        auth = auth.decode() if isinstance(auth, bytes) else auth
        signed_keys.append(auth.split("Credential=")[1].split("/")[0])
        return AWSResponse(
            request.url, 200, HTTPHeaders(), mock.MagicMock(content=b"{}")
        )

    client.meta.events.register("before-send.devicefarm", _capture)
    client.get_run(arn=RUN_ARN)
    client.get_run(arn=RUN_ARN)
    assert signed_keys == ["AKIAFIRST", "AKIASECOND"]


S25_ARN, S26_ARN = (
    aws.get_aws_device_arns(n)[0] for n in ("Samsung Galaxy S25", "Samsung Galaxy S26")
)


def _run(status: str, device_arn: str) -> dict:
    return {
        "status": status,
        "deviceSelectionResult": {"filters": [{"values": [device_arn]}]},
    }


def test_count_waiting_jobs_counts_only_queued_runs_for_the_device() -> None:
    farm = aws.AwsDeviceFarm.__new__(aws.AwsDeviceFarm)
    farm.config = AwsDeviceFarmConfig(project_arn="p")
    farm.client = mock.MagicMock()
    farm.client.list_runs.side_effect = [
        {
            "runs": [
                _run("PENDING_DEVICE", S26_ARN),
                _run("RUNNING", S26_ARN),
                _run("SCHEDULING", S26_ARN),
                _run("PENDING", S25_ARN),
            ],
            "nextToken": "t1",
        },
        {"runs": [_run("PENDING_CONCURRENCY", S26_ARN)], "nextToken": "t2"},
        {"runs": [_run("COMPLETED", S26_ARN)], "nextToken": "t3"},
    ]

    assert farm.count_waiting_jobs("Samsung Galaxy S26") == 3
    # Stops at the first all-completed page instead of walking the whole history.
    assert farm.client.list_runs.call_count == 3


@pytest.mark.parametrize(
    ("job_timeout", "expected"),
    [(None, 600), (0, 0), (900, 900)],
)
def test_aws_submit_bundle_job_timeout_falls_back_to_timeout_only_when_none(
    job_timeout: int | None, expected: int, tmp_path: Path
) -> None:
    farm = aws.AwsDeviceFarm.__new__(aws.AwsDeviceFarm)
    entry = tmp_path / "a.txt"
    entry.write_text("x")
    with (
        mock.patch.object(aws, "get_aws_device_arns", return_value=["arn"]),
        mock.patch.object(farm, "_upload_file", return_value="pkg"),
        mock.patch.object(farm, "_submit_automated_job", return_value="run") as submit,
    ):
        farm.submit_bundle(
            "Samsung Galaxy S25",
            [(str(entry), "a.txt")],
            None,
            "job",
            timeout=600,
            job_timeout=job_timeout,
        )
    assert submit.call_args.kwargs["timeout"] == expected
