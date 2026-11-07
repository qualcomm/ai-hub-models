# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import os
import platform
import posixpath
import shlex
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import adbutils

GENIE_CONFIG_FILE_REL_PATH = "./genie_config.json"

"""
Generating LLM metrics for AI Hub Website

1. Generate model bundles for model of interest
    a. Option A: Use LLM on Genie tutorial https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie
    b. Option B: You already have QNN context binary, tokenizer and QNN SDK in place
2. Use this script to run model on-device using genie-t2t-run and collect data
    a. Example usage:
        python generate_llm_perf_metrics.py --bundle-path <bundle_path_from_point_1> --prompt-file sample_prompt.txt
    b. You can provide `--adb_port <port>` if want to collect data on QDC device with different port than default i.e. 5037
    c. Provide fix prompt to run this multiple iterations
    d. Set `--keep-on-device-bundle` to keep data on-device. This is useful if you plan to use genie-bundle on device for further use.
3. This outputs profiling data in `perf.yaml` format without device info
    e.g.
        ---------------- LLM Metrics ----------------
        Device details: SoC: SM8750 OS: 15

            llm_metrics:
                tokens_per_second: 25.77723

    You can copy paste this as is in new section created in perf.yaml for given device.
"""


def _is_local_support() -> bool:
    return bool(
        "Qualcomm" in platform.processor() and platform.system() in ["Linux", "Windows"]
    )


def _get_processed_kpis(profile: dict) -> tuple[int, float, float | None]:
    """Get Prompt Processor(PP) and Token Generator(TG) metrics from output log"""
    components = profile["components"]
    assert len(components) == 1

    query_events = [
        event
        for event in components[0]["events"]
        if event["type"] == "GenieDialog_query"
    ]
    assert len(query_events) == 1
    ttft = query_events[0]["time-to-first-token"]
    tps = query_events[0]["token-generation-rate"]
    prefill = query_events[0]["prompt-processing-rate"]
    assert ttft["unit"] == "us"
    assert tps["unit"] == "toks/sec"
    assert prefill["unit"] == "toks/sec"
    return ttft["value"], tps["value"], prefill["value"]


def _get_avg_tokens_per_second(token_generator_tps: list[float]) -> float:
    return sum(token_generator_tps) / len(token_generator_tps)


def _get_llm_metrics(
    token_generator_tps: list[float],
    prefill_tps: list[float] | None = None,
    verbose: bool = False,
) -> str:
    """Average Genie's token generation and prompt processing rates."""
    avg_tps = _get_avg_tokens_per_second(token_generator_tps)
    prefill_line = ""
    if prefill_tps:
        avg_prefill_tps = sum(prefill_tps) / len(prefill_tps)
        prefill_line = f"\n        prefill_tokens_per_second: {avg_prefill_tps:.5f}"

    return f"""
    genie:
        context_length: <context_length>
        tokens_per_second: {avg_tps:.5f}{prefill_line}
"""


def _push_bundle_to_device(
    device: adbutils._device.AdbDevice, bundle_path: str, destination_bundle_path: str
) -> None:
    for _, dir_names, file_names in os.walk(bundle_path):
        if len(dir_names) > 0:
            raise RuntimeError(
                "Invalid source bundle directory. Expecting no sub-directory in bundle."
            )

        for file_name in file_names:
            full_file_path = str(os.path.join(bundle_path, file_name))
            device.sync.push(full_file_path, destination_bundle_path)


def _run_model_on_android_device(
    device: adbutils._device.AdbDevice,
    working_dir: str,
    prompt: str | None = None,
    prompt_file: str | None = None,
    num_iterations: int = 5,
    verbose: bool = False,
) -> tuple[list[int], list[float], list[float]]:
    """Runs models on device from working directory"""
    assert (prompt is not None) != (prompt_file is not None)

    local_profile_path = "profile.txt"
    remote_profile_path = posixpath.join(working_dir, "profile.txt")
    prompt_path = None
    if prompt:
        prompt_args = f"-p '{prompt}'"
    else:
        assert prompt_file is not None
        prompt_path = posixpath.join(working_dir, os.path.basename(prompt_file))
        prompt_args = f"--prompt_file '{prompt_path}'"

    device_commands = [
        f"cd {working_dir}",
        f"rm -rf {remote_profile_path}",
        f"export LD_LIBRARY_PATH={working_dir}",
        f"export ADSP_LIBRARY_PATH={working_dir}",
        f"./genie-t2t-run -c {GENIE_CONFIG_FILE_REL_PATH} {prompt_args} --profile {remote_profile_path}",
    ]
    command = ";".join(device_commands)

    if verbose:
        print(f"{command}")

    prompt_processor_time: list[int] = []
    token_generator_tps: list[float] = []
    prefill_tps: list[float] = []
    for i in range(num_iterations):
        print(f"Inference iteration {i + 1}")
        if prompt_file is not None and prompt_path is not None:
            device.sync.push(prompt_file, prompt_path)
        model_output = device.shell(command)
        device.sync.pull(remote_profile_path, local_profile_path)

        with open(local_profile_path) as f:
            profile = json.load(f)
        os.remove(local_profile_path)

        if verbose:
            print("Model output")
            print(model_output)

        # Collect KPIs from model output
        pp_time, tg_tps, pp_rate = _get_processed_kpis(profile)
        prompt_processor_time.append(pp_time)
        token_generator_tps.append(tg_tps)
        if pp_rate is not None:
            prefill_tps.append(pp_rate)

    return prompt_processor_time, token_generator_tps, prefill_tps


def _run_model_on_local_host(
    working_dir: str,
    prompt: str | None = None,
    prompt_file: str | None = None,
    num_iterations: int = 5,
    verbose: bool = False,
) -> tuple[list[int], list[float], list[float]]:
    """Runs models on host machine from working directory"""
    assert (prompt is not None) != (prompt_file is not None)

    if prompt:
        prompt_args = [
            "-p",
            prompt,
        ]
    else:
        assert prompt_file is not None
        prompt_args = ["--prompt_file", prompt_file]

    genie_path = os.path.abspath(os.path.join(working_dir, "genie-t2t-run.exe"))
    if not os.path.exists(genie_path):
        genie_path = "genie-t2t-run"

    genie_config_path = GENIE_CONFIG_FILE_REL_PATH
    profile_full_path = os.path.abspath(os.path.join(working_dir, "profile.txt"))
    if os.path.isfile(profile_full_path):
        os.remove(profile_full_path)
    command = [
        genie_path,
        "-c",
        genie_config_path,
        *prompt_args,
        "--profile",
        profile_full_path,
    ]

    if verbose:
        print(f"From working directory: {working_dir}")
        print(shlex.join(command))

    prompt_processor_time: list[int] = []
    token_generator_tps: list[float] = []
    prefill_tps: list[float] = []
    for i in range(num_iterations):
        print(f"Inference iteration {i + 1}")
        process = subprocess.run(
            command, check=False, cwd=working_dir, capture_output=True, shell=True
        )
        model_output = process.stdout
        if verbose:
            if process.stderr:
                print(f"ERROR: {process.stderr.decode()}")
            else:
                print("Model output")
                print(f"{model_output.decode()}")

        full_path = profile_full_path
        with open(full_path) as f:
            profile = json.load(f)
        os.remove(full_path)

        # Collect KPIs from model output
        pp_time, tg_tps, pp_rate = _get_processed_kpis(profile)
        prompt_processor_time.append(pp_time)
        token_generator_tps.append(tg_tps)
        if pp_rate is not None:
            prefill_tps.append(pp_rate)

    return prompt_processor_time, token_generator_tps, prefill_tps


def main() -> None:
    parser = argparse.ArgumentParser(
        "Generate LLM metrics via Genie by running model on locally or on Android device (via adb)."
    )
    parser.add_argument(
        "--use-adb",
        action="store_true",
        help="Runs the benchmark on a connected Android device.",
    )

    parser.add_argument(
        "--bundle-path",
        type=str,
        default="",
        help="Local path to genie-bundle generated as per https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie",
    )
    parser.add_argument(
        "--device-bundle-path",
        type=str,
        default="",
        help="If provided, uses bundle path from device and skips copying data from --bundle-path",
    )
    parser.add_argument(
        "--keep-on-device-bundle",
        action="store_true",
        help="If set, keeps copied on-device bundled as is. Otherwise, cleans up directory after profiling.",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=10,
        help="Number of times to run Genie inference for model profiling.",
    )
    parser.add_argument(
        "--adb-port",
        type=int,
        default=5037,
        help="Port to connect adb devices on. Usually 5037, but if using QDC devices, match this with device port you set ssh-tunneling to.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Refer to prompt format for given model. Try the prompt first to ensure output is concise to make profiling efficient and fair between models. You can achieve this by adding instructions to be concise in the sytem prompt. E.g., refer to the following prompt for Llama3 models: '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\nYou are a helpful assistant. Answer the question concisely.<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\nWhat is gravity?<|eot_id|><|start_header_id|>assistant<|end_header_id|>'",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        help="Like --prompt, but provide input as a file path relative to the bundle.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, dumps genie model inference output for each inference. Otherwise, minimal output of stdout and final metrics.",
    )

    args = parser.parse_args()
    if (args.prompt is not None) == (args.prompt_file is not None):
        parser.error("Providing either --prompt or --prompt-file is required.")

    bundle_path = args.bundle_path
    device_bundle_path = args.device_bundle_path
    adb_port = args.adb_port
    num_iterations = args.num_iterations
    prompt = args.prompt
    prompt_file = os.path.abspath(args.prompt_file) if args.prompt_file else None
    keep_on_device_bundle = args.keep_on_device_bundle
    verbose = args.verbose

    if bundle_path and device_bundle_path:
        raise RuntimeError(
            "Only one of --bundle-path and --device-bundle-path should be set."
        )

    run_locally = _is_local_support() and not args.use_adb

    if verbose:
        print("-" * 80)
        if run_locally:
            print(f"Bundle path: {bundle_path}")
        else:
            print(f"Device bundle path: {device_bundle_path}")
            print(f"ADB port: {adb_port}")
        print(f"Number of iterations: {num_iterations}")
        if args.prompt_file:
            print(f"Prompt file: {prompt_file}")
        else:
            print(f"Prompt: {prompt}")
        print("-" * 80)

    # Connect to device using ADB client
    if args.use_adb:
        import adbutils

        if verbose:
            print(f"Connecting to Android device via ADB: port {adb_port}")
        adb = adbutils.AdbClient(port=adb_port)
        device = adb.device()
        android_os_version = device.prop.get("ro.system.build.version.release")
        device_os = f"Android OS {android_os_version}"
        device_soc = device.prop.get("ro.soc.model")
    elif run_locally:
        device = None
        device_os = platform.system()
        device_soc = platform.processor()
    else:
        raise RuntimeError(
            "Unable to run locally. Please use --use-adb with connected Android device."
        )

    if bundle_path:
        if not os.path.exists(bundle_path):
            raise RuntimeError(f"Provided --bundle-path {bundle_path} does not exists.")
    elif device_bundle_path:
        if run_locally:
            raise RuntimeError("Please use --bundle-path on Windows")

        assert device is not None
        if device.shell(f"[ -e {device_bundle_path} ]; echo $?") != "0":
            raise RuntimeError(
                f"Provided --device-bundle-path {device_bundle_path} does not exists on device."
            )
    else:
        raise RuntimeError("One of --bundle-path and --device-bundle-path must be set.")

    try:
        if device_bundle_path:
            print("Using bundle from device for profiling.")
            working_dir = device_bundle_path
        elif run_locally:
            working_dir = bundle_path
        else:
            assert device is not None
            working_dir = device.shell("mktemp -d")
            _push_bundle_to_device(device, bundle_path, working_dir)
            print(f"Bundled directory copied onto device at {working_dir}")

        print(f"Measuring perf on {device_os} host with {device_soc}.")

        # Run model on-device using Genie
        extra = dict(num_iterations=num_iterations, verbose=verbose)
        if run_locally:
            ret = _run_model_on_local_host(
                working_dir, prompt=prompt, prompt_file=prompt_file, **extra
            )
        else:
            assert device is not None
            ret = _run_model_on_android_device(
                device, working_dir, prompt=prompt, prompt_file=prompt_file, **extra
            )
        _, token_generator_tps, prefill_tps = ret

        llm_metric = _get_llm_metrics(token_generator_tps, prefill_tps=prefill_tps)

        print("\n---------------- LLM Metrics ----------------")
        print(f"Device details: SoC: {device_soc} OS: {device_os}")
        print(llm_metric)

    finally:
        if keep_on_device_bundle or device_bundle_path or run_locally:
            print(f"Bundle directory from device {working_dir} not removed.")
        else:
            print(f"Removing bundle directory from device {working_dir}.")
            assert device is not None
            device.shell(f"rm -rf {working_dir}")


if __name__ == "__main__":
    main()
