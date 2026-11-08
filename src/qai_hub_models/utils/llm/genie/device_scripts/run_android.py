# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
import subprocess
import sys

import pytest
from appium import webdriver
from appium.options.common import AppiumOptions

options = AppiumOptions()
options.set_capability("automationName", "UiAutomator2")
options.set_capability("platformName", "Android")
options.set_capability("deviceName", os.getenv("ANDROID_DEVICE_VERSION"))
options.set_capability("appium:androidInstallTimeout", 300000)  # 5 minutes
options.set_capability("appium:adbExecTimeout", 300000)  # 5 minutes


def _set_package_verifier(enabled: bool) -> None:
    # Play Protect blocks Appium's unsigned settings_apk-debug.apk install,
    # which surfaces as an adb-install timeout. Toggle both flags off before
    # Appium starts and restore them in teardown.
    value = "1" if enabled else "0"
    for key in ("package_verifier_enable", "verifier_verify_adb_installs"):
        subprocess.run(
            ["adb", "shell", "settings", "put", "global", key, value],
            check=False,
        )


def _preflight_url(url: str) -> None:
    # We've seen QDC SM8750 QRD boot with wifi degraded (logcat shows WifiHAL
    # fatal_event + ENETDOWN), in which case the on-device curls would hang
    # and the test would silently "pass". A presigned S3 URL's signature is
    # bound to the HTTP method, so HEAD gets a 403; `-r 0-0` keeps it a GET.
    preflight = subprocess.run(
        [
            "adb",
            "shell",
            f"curl -sS -o /dev/null -w '%{{http_code}}' --max-time 15 -r 0-0 '{url}'",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    http_code = preflight.stdout.strip()
    if preflight.returncode != 0 or not http_code.startswith(("2", "3")):
        pytest.fail(
            f"Device cannot reach {url.split('?', 1)[0]} "
            f"(rc={preflight.returncode}, http_code={http_code!r}, "
            f"stderr={preflight.stderr!r}). Likely QDC device-side wifi "
            "failure — file a QDC infra ticket and re-run."
        )


class TestGenie:
    @pytest.fixture
    def driver(self) -> webdriver.Remote | None:
        # AWS Device Farm's custom test environment has no Appium session to
        # satisfy (that's a QDC framework formality this test never actually
        # uses `driver` for); skip starting one there.
        if os.environ.get("QAIHM_SKIP_APPIUM"):
            return None
        _set_package_verifier(False)
        try:
            return webdriver.Remote(
                command_executor="http://127.0.0.1:4723/wd/hub", options=options
            )
        finally:
            _set_package_verifier(True)

    def test_genie(self, driver: webdriver.Remote | None) -> None:
        # download qairt sdk via curl on device
        # script to set environment variables
        # run genie-t2t-run on the device
        num_trials = int("<<NUM_TRIALS>>")
        genie_bundle_url = "<<GENIE_BUNDLE_URL>>"
        qairt_sdk_url = "<<QAIRT_SDK_URL>>"
        trial_commands = []
        for i in range(num_trials):
            trial_commands.append(
                f'sed -i \'s/"seed": [0-9]*/"seed": {i}/\' genie_config.json'
            )
            trial_commands.append(
                f"genie_retry genie-t2t-run -c genie_config.json --prompt_file sample_prompt.txt --profile /data/local/tmp/<<DEVICE_LOGS_DIR>>/profile{i}.json 2>>/data/local/tmp/<<DEVICE_LOGS_DIR>>/genie_stderr.log"
            )
        full_genie_command = " && ".join(trial_commands)
        qairt_path = "/data/local/tmp/qairt/<<QAIRT_VERSION>>"
        if genie_bundle_url:
            genie_bundle_download_block = f"""# Always re-download: dedicated-pool devices are reused, so a partial extract
# from a previous job would otherwise silently corrupt this run.
rm -rf /data/local/tmp/genie_bundle /data/local/tmp/_genie_dl
mkdir -p /data/local/tmp/_genie_dl
curl -L --fail --max-time 1800 --retry 3 --retry-delay 5 \\
    --output /data/local/tmp/genie_bundle.zip "{genie_bundle_url}"
unzip -q /data/local/tmp/genie_bundle.zip -d /data/local/tmp/_genie_dl
rm -f /data/local/tmp/genie_bundle.zip
GENIE_EXTRACTED=$(find /data/local/tmp/_genie_dl -mindepth 1 -maxdepth 1 -type d | head -n1)
mv "$GENIE_EXTRACTED" /data/local/tmp/genie_bundle
rm -rf /data/local/tmp/_genie_dl
if [ -d /data/local/tmp/eval_prompts ]; then
    mv /data/local/tmp/eval_prompts /data/local/tmp/genie_bundle/prompts
fi"""
        else:
            # Carve-out fallback: genie_bundle (weights included) was already
            # pushed host-side via adb push, nothing to download.
            genie_bundle_download_block = ""
        genie_script = f"""set -e
# We pipe genie output through `tee` (below) so it shows up on adb stdout
# (and thus in the captured proc.stdout) even when a failed QDC job never
# makes the on-device log files available. pipefail keeps the pipeline's
# exit status tied to genie rather than to tee, which always succeeds.
set -o pipefail
# Drop per-job state on exit (dedicated-pool devices are reused).
cleanup_device() {{
    rm -rf /data/local/tmp/genie_bundle \\
           /data/local/tmp/genie_bundle.zip \\
           /data/local/tmp/_genie_dl \\
           /data/local/tmp/eval_prompts \\
           /data/local/tmp/qairt \\
           /data/local/tmp/qairt.zip 2>/dev/null || true
}}
trap cleanup_device EXIT
# genie-t2t-run fails randomly on QDC devices; give each invocation one retry
# before letting the failure (and set -e) abort the whole job. Redirect stderr
# to a log file: QDC flags jobs Unsuccessful on any stderr output (PR #3641).
genie_retry() {{
    tmp_out=$(mktemp)
    if ! "$@" | tee "$tmp_out"; then
        if grep -q "Context Size was exhausted" "$tmp_out"; then
            echo "genie_retry: context size exhausted, skipping retry: $*" >&2
        else
            echo "genie_retry: command failed, retrying once: $*" >&2
            "$@"
        fi
    fi
    rm -f "$tmp_out"
}}
{genie_bundle_download_block}
cd /data/local/tmp/genie_bundle
rm -rf /data/local/tmp/qairt
echo "=== Pre-download connectivity check ==="
echo "Pinging google.com before QAIRT SDK download..."
ping -c 1 google.com && echo "Pre-download ping: SUCCESS" || echo "Pre-download ping: FAILED"
curl -L --fail --max-time 3600 --retry 3 --retry-delay 5 --output /data/local/tmp/qairt.zip "{qairt_sdk_url}"
echo "=== Post-download connectivity check ==="
echo "Pinging google.com after QAIRT SDK download..."
ping -c 1 google.com && echo "Post-download ping: SUCCESS" || echo "Post-download ping: FAILED"
unzip -q /data/local/tmp/qairt.zip -d /data/local/tmp || {{
    echo "unzip failed, retrying once" >&2
    rm -rf /data/local/tmp/qairt
    unzip -q /data/local/tmp/qairt.zip -d /data/local/tmp
}}
export QAIRT_HOME={qairt_path}
export PATH={qairt_path}/bin/aarch64-android:${{PATH}}
export LD_LIBRARY_PATH={qairt_path}/lib/aarch64-android
export ADSP_LIBRARY_PATH={qairt_path}/lib/hexagon-<<HEXAGON_VERSION>>/unsigned

# Drop stale logs from a prior job on this shared device.
rm -rf /data/local/tmp/<<DEVICE_LOGS_DIR>>
mkdir -p /data/local/tmp/<<DEVICE_LOGS_DIR>>
# Cap generation for perf so a diverging model can't stall profiling.
sed -i 's/"max-num-tokens": [0-9]*/"max-num-tokens": 128/' genie_config.json
genie_retry genie-t2t-run -c genie_config.json --prompt_file sample_prompt.txt 2>>/data/local/tmp/<<DEVICE_LOGS_DIR>>/genie_stderr.log | tee /data/local/tmp/<<DEVICE_LOGS_DIR>>/genie.log
{full_genie_command}

PROMPT_DIR=/data/local/tmp/genie_bundle/prompts
EVAL_OUTPUT_FILE=/data/local/tmp/<<DEVICE_LOGS_DIR>>/eval_outputs.txt
if [ -d "$PROMPT_DIR" ]; then
    # The perf sweep above left the seed at {num_trials - 1}; reset to the
    # bundle's nominal default so eval isn't silently run at a different seed.
    sed -i 's/"seed": [0-9]*/"seed": 42/' genie_config.json
    sed -i 's/"max-num-tokens": [0-9]*/"max-num-tokens": 2048/' genie_config.json
    # Switch to power_saver perf_profile: sustained burst thermal-throttles and kills the eval loop on QDC SM8750.
    sed -i 's/"perf_profile": "[^"]*"/"perf_profile": "power_saver"/' htp_backend_ext_config.json
    > "$EVAL_OUTPUT_FILE"
    for prompt_file in $PROMPT_DIR/prompt_*.txt; do
        idx=$(basename "$prompt_file" | sed 's/prompt_\\([0-9]*\\)\\.txt/\\1/')
        echo "===EVAL_IDX_${{idx}}===" | tee -a "$EVAL_OUTPUT_FILE"
        genie_retry genie-t2t-run -c genie_config.json --prompt_file "$prompt_file" 2>&1 | tee -a "$EVAL_OUTPUT_FILE"
        # Short inter-prompt cooldown to keep the HTP from thermal-throttling.
        sleep 3
    done
fi
"""
        # QDC stages the extracted test package under /qdc/appium; AWS Device
        # Farm's custom test environment extracts it under
        # $DEVICEFARM_TEST_PACKAGE_PATH instead (see aws_test_spec.yaml,
        # which sets the override).
        host_artifact_root = os.environ.get("QAIHM_HOST_ARTIFACT_ROOT", "/qdc/appium")
        if genie_bundle_url:
            # The genie_bundle itself is curled directly on-device (see
            # genie_script below); only the small eval-prompt files (if any)
            # are staged host-side and need pushing.
            eval_prompts_src = f"{host_artifact_root}/eval_prompts"
            if os.path.isdir(eval_prompts_src):
                subprocess.run(
                    ["adb", "push", f"{eval_prompts_src}/", "/data/local/tmp"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
        else:
            # Carve-out fallback (e.g. no presigned URL available): push the
            # full genie_bundle directory, weights included, as before.
            subprocess.run(
                [
                    "adb",
                    "push",
                    f"{host_artifact_root}/genie_bundle/",
                    "/data/local/tmp",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

        _preflight_url(qairt_sdk_url)
        if genie_bundle_url:
            _preflight_url(genie_bundle_url)

        # Run the shell script on the device. adb shell does not propagate the
        # remote exit code, so on-device failures can't be detected here; the
        # output-existence check below is what catches them.
        proc = subprocess.run(
            ["adb", "shell", "sh", "-c", genie_script],
            capture_output=True,
            text=True,
            check=True,  # only catches adb-side failures, not on-device ones
        )

        # Since adb shell hides the on-device exit code, confirm the script
        # actually produced its outputs. A green pytest with no genie.log was
        # the failure mode on QDC job 613912.
        expected = ["/data/local/tmp/<<DEVICE_LOGS_DIR>>/genie.log"] + [
            f"/data/local/tmp/<<DEVICE_LOGS_DIR>>/profile{i}.json"
            for i in range(num_trials)
        ]
        ls = subprocess.run(
            ["adb", "shell", "ls", "-l", *expected],
            check=False,
            capture_output=True,
            text=True,
        )
        if ls.returncode != 0:
            pytest.fail(
                "Expected on-device outputs are missing — the genie script "
                "likely failed on device.\n"
                f"--- ls stdout ---\n{ls.stdout}\n--- ls stderr ---\n{ls.stderr}\n"
                f"--- script stdout ---\n{proc.stdout}\n"
                f"--- script stderr ---\n{proc.stderr}"
            )


if __name__ == "__main__":
    # Invoke Pytest on this file
    sys.exit(pytest.main(["-s", "--junitxml=results.xml", os.path.realpath(__file__)]))
