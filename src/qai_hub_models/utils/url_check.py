# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Shared HTTP reachability checker for manifest URLs.

Extracted from ``qai_hub_models.configs.manifest_yaml`` so both the
in-tree lint (``test_manifest_yamls``) and the ``qai-hub-models validate``
CLI can share the same 24 h JSON cache and retry policy.

Checks start with HEAD. A 403 or 405 falls back to a streaming GET whose
body is never read, because some hosts reject HEAD while still serving
the resource.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from qai_hub_models_cli.common import LOCAL_STORE_DEFAULT_PATH
from urllib3.util.retry import Retry

URL_CACHE_TTL_SECONDS = 86400
URL_CACHE_PATH = Path(LOCAL_STORE_DEFAULT_PATH) / "url_check_cache.json"

# URLs that should be skipped during validation (e.g., due to SSL issues or
# rate limiting in CI environments).
URL_VALIDATION_SKIPLIST = {
    # llama_v3_taide_8b_chat license - SSL cert issue
    "https://drive.google.com/file/d/1ICTxogjS9Bc2O3K1P9ZauQYVoruT13n5/view?pli=1",
    # indus_1b research paper - intermittent 403
    "https://www.techmahindra.com/makers-lab/indus-project/",
    # llama research papers - blocked by the corporate firewall in CI
    "https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/",
    "https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/",
}

# HEAD statuses that mean the resource is reachable. 202 is what IEEE returns
# to HEAD; 429 is treated as reachable so a rate limit is not a missing asset.
_REACHABLE_STATUSES = (
    requests.codes.ok,
    requests.codes.accepted,
    requests.codes.too_many_requests,
)

# Hosts that reject HEAD but still serve GET. qwen.ai answers HEAD with 405;
# 403 is the same shape from some CDNs. Confirm with a body-less GET.
_HEAD_FALLBACK_STATUSES = (
    requests.codes.forbidden,
    requests.codes.method_not_allowed,
)


def _load_url_cache() -> dict[str, float]:
    if not URL_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(URL_CACHE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_url_cache(cache: dict[str, float]) -> None:
    URL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    URL_CACHE_PATH.write_text(json.dumps(cache))


def _make_url_check_session() -> requests.Session:
    """Create a Session that retries on 502/503/504 and connection failures."""
    # S3 throttles a burst of HEADs with 503; without these in the forcelist a
    # transient server error is reported as a missing asset.
    retry = Retry(total=4, backoff_factor=1, status_forcelist=[502, 503, 504])
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def head_check_urls(urls: list[tuple[str, str]]) -> list[str]:
    """HEAD-check a list of ``(url, error_label)`` pairs in parallel.

    URLs that were successfully checked within the last 24 hours are
    skipped, as are URLs on the validation skiplist. Returns the list of
    failure messages (empty on success).

    HEAD 200, 202, and 429 are accepted as reachable. HEAD 403 and 405 are
    not final: the same session then issues a streaming GET with the same
    headers, redirects, 10-second timeout, and 502/503/504 retry policy.
    The GET body is never read. GET 200, 202, and 429 are accepted for
    compatibility with hosts that reject HEAD; GET 403, 405, any other
    status, and GET request exceptions fail. HEAD request exceptions do
    not fall back to GET. Only a successful final outcome is cached.

    Parameters
    ----------
    urls
        List of ``(url, label)`` pairs. ``label`` is the human-readable
        description used in the error message.

    Returns
    -------
    list[str]
        Error messages for each URL that failed. Empty if all URLs are
        reachable (or in the skiplist / cache).
    """
    if not urls:
        return []

    now = time.time()
    cache = _load_url_cache()
    urls_to_check = [
        (url, label)
        for url, label in urls
        if url not in URL_VALIDATION_SKIPLIST
        and now - cache.get(url, 0) > URL_CACHE_TTL_SECONDS
    ]

    if not urls_to_check:
        return []

    session = _make_url_check_session()
    # IEEE rejects HEAD requests that arrive without a User-Agent (error 418).
    headers = {"User-Agent": "QAIHM Test Suite"}

    def _check(url: str, label: str) -> str | None:
        try:
            head_response = session.head(
                url, allow_redirects=True, timeout=10, headers=headers
            )
        except requests.RequestException as e:
            # A HEAD transport failure is final. Do not fall back to GET.
            return f"{label} at {url} ({e})"
        try:
            status = head_response.status_code
        except requests.RequestException as e:
            return f"{label} at {url} ({e})"
        finally:
            # Release the HEAD connection before a possible GET fallback.
            head_response.close()

        # 403/405 often means the host blocks HEAD (qwen.ai returns 405)
        # while GET still serves the resource. Stream the GET and do not
        # read the body.
        if status in _HEAD_FALLBACK_STATUSES:
            try:
                with session.get(
                    url,
                    allow_redirects=True,
                    timeout=10,
                    headers=headers,
                    stream=True,
                ) as get_response:
                    status = get_response.status_code
            except requests.RequestException as e:
                return f"{label} at {url} ({e})"

        if status not in _REACHABLE_STATUSES:
            return f"{label} at {url} (status: {status})"
        cache[url] = now
        return None

    with ThreadPoolExecutor(max_workers=len(urls_to_check)) as pool:
        results = list(pool.map(lambda t: _check(*t), urls_to_check))
    errors = [r for r in results if r is not None]

    _save_url_cache(cache)
    return errors


def validate_urls_exist(urls: list[tuple[str, str]]) -> None:
    """Check URLs and raise ``ValueError`` on failure.

    Thin wrapper around :func:`head_check_urls` kept for callers (like the
    old manifest_yaml validator) that expect an exception-based interface.
    HEAD 403 and 405 fall back to a streaming GET before the URL is failed.
    """
    errors = head_check_urls(urls)
    if errors:
        raise ValueError("\n".join(errors))
