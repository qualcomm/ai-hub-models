# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests
from requests.adapters import HTTPAdapter
from typing_extensions import Self

from qai_hub_models.utils import url_check
from qai_hub_models.utils.url_check import _make_url_check_session

_URL = "https://example.invalid/asset"
_LABEL = "Asset"
_HEADERS = {"User-Agent": "QAIHM Test Suite"}


class _Response:
    """HEAD/GET stand-in that records close, context exit, and body access."""

    def __init__(
        self, status_code: int, status_error: BaseException | None = None
    ) -> None:
        self._status_code = status_code
        self._status_error = status_error
        self.closed = False
        self.exited = False
        self.body_reads = 0
        self.raw = _Raw(self)

    @property
    def status_code(self) -> int:
        if self._status_error is not None:
            raise self._status_error
        return self._status_code

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.exited = True
        self.close()

    @property
    def content(self) -> bytes:
        self.body_reads += 1
        return b""

    @property
    def text(self) -> str:
        self.body_reads += 1
        return ""

    def json(self) -> dict[str, object]:
        self.body_reads += 1
        return {}

    def iter_content(self, *_args: object, **_kwargs: object) -> object:
        self.body_reads += 1
        return iter(())


class _Raw:
    def __init__(self, response: _Response) -> None:
        self._response = response

    def read(self, *_args: object, **_kwargs: object) -> bytes:
        self._response.body_reads += 1
        return b""


def _isolate_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    cache_path = tmp_path / "url_check_cache.json"
    monkeypatch.setattr(url_check, "URL_CACHE_PATH", cache_path)
    return cache_path


def _use_session(monkeypatch: pytest.MonkeyPatch, session: MagicMock) -> None:
    monkeypatch.setattr(url_check, "_make_url_check_session", lambda: session)


def _assert_probe(call: MagicMock, *, stream: bool) -> None:
    assert call.call_count == 1
    args, kwargs = call.call_args
    assert args == (_URL,)
    assert kwargs["allow_redirects"] is True
    assert kwargs["timeout"] == 10
    assert kwargs["headers"] == _HEADERS
    if stream:
        assert kwargs["stream"] is True
    else:
        assert "stream" not in kwargs


def _saved(cache_path: Path) -> dict[str, float]:
    return json.loads(cache_path.read_text())


def test_transient_server_errors_are_retried() -> None:
    """A throttled or briefly unavailable host must not read as a missing asset.

    S3 answered a HEAD burst with 503 and six models' banners were reported
    absent; 503/504 belong in the forcelist alongside 502. The GET fallback
    uses this same adapter, so GET stays retry-eligible.
    """
    adapter = _make_url_check_session().get_adapter("https://example.com")
    assert isinstance(adapter, HTTPAdapter)
    retry = adapter.max_retries
    assert {502, 503, 504} <= set(retry.status_forcelist or [])
    allowed = getattr(retry, "allowed_methods", None)
    assert allowed is None or "GET" in allowed
    assert allowed is None or "HEAD" in allowed


@pytest.mark.parametrize("status", [200, 202, 429])
def test_accepted_head_does_not_get(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: int
) -> None:
    cache_path = _isolate_cache(monkeypatch, tmp_path)
    session = MagicMock()
    head_resp = _Response(status)
    session.head.return_value = head_resp
    _use_session(monkeypatch, session)

    assert url_check.head_check_urls([(_URL, _LABEL)]) == []

    _assert_probe(session.head, stream=False)
    session.get.assert_not_called()
    assert head_resp.closed
    assert head_resp.body_reads == 0
    assert isinstance(_saved(cache_path)[_URL], float)


@pytest.mark.parametrize("status", [404, 500])
def test_other_head_failures_do_not_fall_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: int
) -> None:
    cache_path = _isolate_cache(monkeypatch, tmp_path)
    session = MagicMock()
    head_resp = _Response(status)
    session.head.return_value = head_resp
    _use_session(monkeypatch, session)

    errors = url_check.head_check_urls([(_URL, _LABEL)])

    assert errors == [f"{_LABEL} at {_URL} (status: {status})"]
    _assert_probe(session.head, stream=False)
    session.get.assert_not_called()
    assert head_resp.closed
    assert head_resp.body_reads == 0
    assert _URL not in _saved(cache_path)


@pytest.mark.parametrize(
    "exc",
    [
        requests.ConnectionError("no route"),
        requests.Timeout("timed out"),
    ],
)
def test_head_request_exceptions_do_not_fall_back(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    exc: requests.RequestException,
) -> None:
    cache_path = _isolate_cache(monkeypatch, tmp_path)
    session = MagicMock()
    session.head.side_effect = exc
    _use_session(monkeypatch, session)

    errors = url_check.head_check_urls([(_URL, _LABEL)])

    assert errors == [f"{_LABEL} at {_URL} ({exc})"]
    session.get.assert_not_called()
    assert _URL not in _saved(cache_path)


@pytest.mark.parametrize("head_status", [403, 405])
@pytest.mark.parametrize(
    ("get_outcome", "expect_failure"),
    [
        pytest.param(200, False, id="get-200"),
        pytest.param(202, False, id="get-202"),
        pytest.param(429, False, id="get-429"),
        pytest.param(403, True, id="get-403"),
        pytest.param(404, True, id="get-404"),
        pytest.param(405, True, id="get-405"),
        pytest.param(500, True, id="get-500"),
        pytest.param(requests.Timeout("timed out"), True, id="get-timeout"),
        pytest.param(
            requests.ConnectionError("connection failed"), True, id="get-connection"
        ),
    ],
)
def test_head_denied_falls_back_to_streaming_get(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    head_status: int,
    get_outcome: int | requests.RequestException,
    expect_failure: bool,
) -> None:
    cache_path = _isolate_cache(monkeypatch, tmp_path)
    session = MagicMock()
    head_resp = _Response(head_status)
    session.head.return_value = head_resp
    get_resp: _Response | None = None
    if isinstance(get_outcome, int):
        get_resp = _Response(get_outcome)

        def _get(*_args: object, **_kwargs: object) -> _Response:
            assert head_resp.closed
            assert get_resp is not None
            return get_resp

        session.get.side_effect = _get
    else:

        def _get_error(*_args: object, **_kwargs: object) -> None:
            assert head_resp.closed
            raise get_outcome

        session.get.side_effect = _get_error
    _use_session(monkeypatch, session)

    errors = url_check.head_check_urls([(_URL, _LABEL)])

    _assert_probe(session.head, stream=False)
    _assert_probe(session.get, stream=True)
    assert head_resp.closed
    assert head_resp.body_reads == 0
    saved = _saved(cache_path)
    if expect_failure:
        if isinstance(get_outcome, int):
            assert errors == [f"{_LABEL} at {_URL} (status: {get_outcome})"]
        else:
            assert errors == [f"{_LABEL} at {_URL} ({get_outcome})"]
        assert _URL not in saved
    else:
        assert errors == []
        assert isinstance(saved[_URL], float)
    if get_resp is not None:
        assert get_resp.exited
        assert get_resp.closed
        assert get_resp.body_reads == 0


def test_warm_cache_and_skiplist_make_no_requests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_path = _isolate_cache(monkeypatch, tmp_path)
    cache_path.write_text(json.dumps({_URL: time.time()}))

    def _forbid() -> MagicMock:
        raise AssertionError("session created")

    monkeypatch.setattr(url_check, "_make_url_check_session", _forbid)
    assert url_check.head_check_urls([(_URL, _LABEL)]) == []

    monkeypatch.setattr(url_check, "URL_VALIDATION_SKIPLIST", {_URL})
    cache_path.unlink()
    assert url_check.head_check_urls([(_URL, _LABEL)]) == []


def test_expired_cache_entry_is_checked_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_path = _isolate_cache(monkeypatch, tmp_path)
    stale = time.time() - url_check.URL_CACHE_TTL_SECONDS - 5
    cache_path.write_text(json.dumps({_URL: stale}))
    session = MagicMock()
    session.head.return_value = _Response(200)
    _use_session(monkeypatch, session)

    assert url_check.head_check_urls([(_URL, _LABEL)]) == []

    session.head.assert_called_once()
    session.get.assert_not_called()
    assert _saved(cache_path)[_URL] > stale


def test_get_context_closes_when_status_read_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A GET that fails while the body is still unread must leave the context."""
    cache_path = _isolate_cache(monkeypatch, tmp_path)
    session = MagicMock()
    head_resp = _Response(403)
    get_resp = _Response(0, status_error=requests.Timeout("slow body"))
    session.head.return_value = head_resp
    session.get.return_value = get_resp
    _use_session(monkeypatch, session)

    errors = url_check.head_check_urls([(_URL, _LABEL)])

    assert errors == [f"{_LABEL} at {_URL} (slow body)"]
    assert head_resp.closed
    assert get_resp.exited
    assert get_resp.closed
    assert get_resp.body_reads == 0
    assert _URL not in _saved(cache_path)
