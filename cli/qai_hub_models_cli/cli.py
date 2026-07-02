# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import functools
import sys
from collections.abc import Callable, Iterable
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from packaging.version import Version
from prettytable import PrettyTable

from qai_hub_models_cli._internal.utils import is_internal_repo, use_internal_releases
from qai_hub_models_cli.args import (
    RUNTIME_VALUES,
    add_asset_filter_args,
    add_chipset_attribute_filter_args,
    add_model_metric_filter_args,
    add_quiet_arg,
    add_version_arg,
    flatten_multi_arg,
    parse_version_arg,
)
from qai_hub_models_cli.common import (
    AIHUB_MODELS_URL,
    CLI_NAME,
    build_filter_command,
    format_command_sections,
    is_heavy_package_installed,
    model_repo_url,
    parse_sdk_version_filters,
    sample_command,
)
from qai_hub_models_cli.envvars import (
    VERBOSE_EXCEPTIONS_ENVVAR,
    bool_envvar_value,
)
from qai_hub_models_cli.fetch import fetch, get_asset_url
from qai_hub_models_cli.find import find_matching_releases
from qai_hub_models_cli.proto.info_pb2 import (
    MODEL_TAG_LLM,
    MODEL_TAG_VLM,
    ModelDomain,
    ModelTag,
    ModelUseCase,
)
from qai_hub_models_cli.proto.manifest_pb2 import ManifestModelEntry
from qai_hub_models_cli.proto.platform_pb2 import FormFactor, WebsiteWorld
from qai_hub_models_cli.proto_helpers.info import get_model_info
from qai_hub_models_cli.proto_helpers.manifest import get_manifest, get_manifest_entry
from qai_hub_models_cli.proto_helpers.numerics import (
    filter_numerics,
    format_numerics_table,
    get_model_numerics,
)
from qai_hub_models_cli.proto_helpers.perf import (
    filter_perf,
    format_perf_table,
    get_model_perf,
)
from qai_hub_models_cli.proto_helpers.platform import (
    filter_chipsets,
    filter_devices,
    format_chipsets_table,
    format_devices_table,
    format_runtime_links,
    format_runtimes_table,
    get_platform,
    resolve_chipset,
)
from qai_hub_models_cli.proto_helpers.platform_enums import (
    domain_proto_to_str,
    form_factor_proto_to_str,
    form_factor_str_to_proto,
    license_proto_to_str,
    normalize_label,
    os_str_to_proto,
    runtime_proto_to_str,
    runtime_str_to_proto,
    tag_proto_to_str,
    tag_str_to_proto,
    use_case_proto_to_str,
    use_case_str_to_proto,
    world_proto_to_str,
    world_str_to_proto,
)
from qai_hub_models_cli.proto_helpers.release_assets import (
    filter_release_assets,
    format_fetch_commands,
    format_release_assets_table,
    get_model_asset_details,
    get_model_release_assets,
)
from qai_hub_models_cli.proto_helpers.tool_versions import format_tool_versions
from qai_hub_models_cli.utils import build_table, wrap_table_column
from qai_hub_models_cli.versions import (
    CURRENT_VERSION,
    MIN_MODEL_FILTER_VERSION,
    UnsupportedVersionError,
    feature_supported,
    get_supported_versions,
    print_upgrade_notice,
    version_flag,
)


def _check_version_match() -> None:
    """Exit if qai_hub_models and qai_hub_models_cli versions differ."""
    try:
        cli_version = version("qai_hub_models_cli")
        models_version = version("qai_hub_models")
    except PackageNotFoundError:
        return
    if cli_version != models_version:
        print(
            f"Version mismatch: qai_hub_models_cli=={cli_version} "
            f"but qai_hub_models=={models_version}. "
            "Please reinstall both packages from the same version."
        )
        sys.exit(1)


def _run_fetch(args: argparse.Namespace) -> None:
    sdk_versions = parse_sdk_version_filters(args.sdk_version or [])

    if args.info:
        all_assets = get_model_release_assets(args.model, args.qaihm_version)
        platform = get_platform(args.qaihm_version)
        release_assets = filter_release_assets(
            all_assets,
            platform,
            args.runtime,
            args.precision,
            args.chipset,
            args.device,
            sdk_versions,
        )
        if not release_assets.assets:
            print("No release assets match the given filters.")
            return
        print(
            format_release_assets_table(
                release_assets,
                platform.chipsets,
                title="Download Options",
                runtimes=platform.runtimes,
            )
        )
        print()
        print(
            format_fetch_commands(
                release_assets,
                args.model,
                # The user is already running -i, so don't suggest it again.
                subset=False,
                runtime=args.runtime,
                precision=args.precision,
                chipset=args.chipset,
                device=args.device,
                sdk_versions=sdk_versions,
                version=args.qaihm_version,
            )
        )
        return

    try:
        if args.url_only:
            url = get_asset_url(
                model=args.model,
                runtime=args.runtime,
                precision=args.precision,
                version=args.qaihm_version,
                chipset=args.chipset,
                device=args.device,
                quiet=args.quiet,
                url_only=True,
                sdk_versions=sdk_versions,
            )
            print(url)
            return

        result = fetch(
            model=args.model,
            runtime=args.runtime,
            precision=args.precision,
            chipset=args.chipset,
            device=args.device,
            version=args.qaihm_version,
            extract=args.extract,
            output_dir=args.output_dir,
            quiet=args.quiet,
            sdk_versions=sdk_versions,
        )
    except Exception as e:
        if args.quiet and not isinstance(
            e, (FileNotFoundError, UnsupportedVersionError)
        ):
            print(
                "Failed to fetch model. Consider excluding -q/--quiet from your command to reveal more logs."
            )
        raise

    result = result.resolve()

    if args.quiet:
        print(result)
        return

    if args.extract:
        print(f"Extracted to: {result}")
    else:
        print(f"Saved to: {result}")

    try:
        asset = get_model_asset_details(
            get_model_release_assets(args.model, args.qaihm_version),
            get_platform(args.qaihm_version),
            args.runtime,
            args.precision,
            args.chipset,
            args.device,
        )
    except Exception:
        asset = None
    if asset is not None and asset.HasField("tool_versions"):
        print(
            f"\nThis download was verified with: "
            f"{format_tool_versions(asset.tool_versions)}\n"
            "Run the model with matching versions to match our reported numerics and performance. Other "
            "versions may behave differently or fail to run."
        )

    print_upgrade_notice()


def add_fetch_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "fetch",
        help="Download a pre-compiled model asset.",
    )
    parser.add_argument("model", type=str.lower, help="Model ID (e.g. mobilenet_v2).")
    add_asset_filter_args(parser)
    add_version_arg(parser)
    parser.add_argument(
        "--extract",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extract the downloaded zip archive (default: true). Use --no-extract to skip.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="Output directory. Default: current directory.",
    )
    parser.add_argument(
        "--url-only",
        action="store_true",
        help="Print the download URL only (do not download).",
    )
    parser.add_argument(
        "-i",
        "--info",
        action="store_true",
        help="List the supported release assets and return without downloading. "
        "The runtime, precision, chipset, device, and sdk-version args act as filters.",
    )
    add_quiet_arg(parser, "Suppress all output except the result path.")
    parser.set_defaults(func=_run_fetch)
    return parser


def _run_find(args: argparse.Namespace) -> None:
    sdk_versions = parse_sdk_version_filters(args.sdk_version or [])

    # Search newest-first; print a progress line per release unless quiet.
    def _progress(version: Version, found: bool, skip_reason: str | None) -> None:
        if found:
            suffix = " (match)"
        elif skip_reason:
            suffix = f" ({skip_reason})"
        else:
            suffix = " (no matching asset)"
        print(f"Searching v{version}...{suffix}", file=sys.stderr)

    results = find_matching_releases(
        args.model,
        runtime=args.runtime,
        precision=args.precision,
        chipset=args.chipset,
        device=args.device,
        sdk_versions=sdk_versions,
        min_version=args.min_version,
        max_version=args.max_version,
        first_only=not args.all,
        progress=None if args.quiet else _progress,
    )

    if not results:
        print("\nCould not find a release with an asset matching the given filters.")
        return

    if args.quiet:
        for version, _ in results:
            print(version)
        return

    for version, release_assets in results:
        platform = get_platform(version)
        print(
            format_release_assets_table(
                release_assets,
                platform.chipsets,
                title=f"Matching Assets (v{version})",
                runtimes=platform.runtimes,
            )
        )
        print()
        print(
            format_fetch_commands(
                release_assets,
                args.model,
                subset=False,
                runtime=args.runtime,
                precision=args.precision,
                chipset=args.chipset,
                device=args.device,
                sdk_versions=sdk_versions,
                include_info=True,
                version=version,
            )
        )
        print()


def add_find_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "find",
        help="Search past releases for an asset matching the given filters.",
        description="Search released versions (newest first) for a model asset "
        "matching the same filters accepted by `fetch`, and report the release(s) "
        "that have one. Useful when the current release dropped an asset you need. "
        "By default only the newest matching release is reported.",
    )
    parser.add_argument("model", type=str.lower, help="Model ID (e.g. mobilenet_v2).")
    add_asset_filter_args(parser)
    parser.add_argument(
        "--min-version",
        default=None,
        type=parse_version_arg,
        help="Only search releases at or above this version (e.g. 0.52.0).",
    )
    parser.add_argument(
        "--max-version",
        default=None,
        type=parse_version_arg,
        help="Only search releases at or below this version (e.g. 0.55.0).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Report every matching release, not just the newest.",
    )
    add_quiet_arg(
        parser,
        "Print matching release versions only, one per line, with no progress output.",
    )
    parser.set_defaults(func=_run_find)
    return parser


def _print_model_metric_footer(command: str, args: argparse.Namespace) -> None:
    """Print the related-command hints shown beneath a perf/numerics table.

    Points at the listing commands for the dimensions a user can filter on, plus
    an example ``command`` invocation pre-filled with the filters already passed
    (placeholders for the rest).
    """
    vflag = version_flag(args.qaihm_version)
    filter_cmd = build_filter_command(
        command,
        args.model,
        vflag,
        runtimes=flatten_multi_arg(args.runtime),
        precisions=flatten_multi_arg(args.precision),
        chipsets=flatten_multi_arg(args.chipset),
        devices=flatten_multi_arg(args.device),
    )
    # The component filter is perf-only; sdk-version applies to both commands.
    for comp in flatten_multi_arg(getattr(args, "component", None)) or []:
        filter_cmd += f" --component '{comp}'"
    for query in args.sdk_version or []:
        filter_cmd += f" -s '{query}'"

    # Cross-link to the sibling metric command (perf <-> numerics).
    sibling = "numerics" if command == "perf" else "perf"
    sibling_label = (
        "Accuracy metrics" if sibling == "numerics" else "Performance metrics"
    )

    print()
    print(
        format_command_sections(
            {
                "Platform Info": [
                    ("More about runtimes", sample_command("runtimes", vflag)),
                    ("Chipset information", sample_command("chipsets", vflag)),
                    ("See devices per chipset", sample_command("devices", vflag)),
                ],
                "Model Info": [
                    ("Full model details", sample_command("info", args.model, vflag)),
                    (sibling_label, sample_command(sibling, args.model, vflag)),
                    ("Filter these results", filter_cmd),
                ],
            }
        )
    )


def _run_perf(args: argparse.Namespace) -> None:
    sdk_versions = parse_sdk_version_filters(args.sdk_version or [])
    platform = get_platform(args.qaihm_version)
    perf = get_model_perf(args.model, args.qaihm_version)
    perf = filter_perf(
        perf,
        platform,
        runtime=flatten_multi_arg(args.runtime),
        precision=flatten_multi_arg(args.precision),
        chipset=flatten_multi_arg(args.chipset),
        device=flatten_multi_arg(args.device),
        sdk_versions=sdk_versions,
        components=flatten_multi_arg(args.component),
    )
    print(format_perf_table(perf, platform=platform))
    if perf.performance_metrics:
        _print_model_metric_footer("perf", args)


def add_perf_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "perf",
        help="Show a model's performance metrics.",
        description="Display per-device performance metrics (inference time, "
        "memory, compute unit) for a model. The runtime, precision, chipset, "
        "device, and sdk-version args act as filters.",
    )
    parser.add_argument(
        "model", type=str.lower, help="Model ID or display name (e.g. mobilenet_v2)."
    )
    add_model_metric_filter_args(parser)
    parser.add_argument(
        "--component",
        nargs="+",
        action="append",
        default=None,
        help="Filter to the given component(s), for multi-component models. "
        "May be repeated or given multiple values.",
    )
    add_version_arg(parser)
    parser.set_defaults(func=_run_perf)
    return parser


def _run_numerics(args: argparse.Namespace) -> None:
    platform = get_platform(args.qaihm_version)
    numerics = get_model_numerics(args.model, args.qaihm_version)
    numerics = filter_numerics(
        numerics,
        platform,
        runtime=flatten_multi_arg(args.runtime),
        precision=flatten_multi_arg(args.precision),
        chipset=flatten_multi_arg(args.chipset),
        device=flatten_multi_arg(args.device),
        sdk_versions=parse_sdk_version_filters(args.sdk_version or []),
    )
    print(format_numerics_table(numerics, runtimes=platform.runtimes))
    if numerics.metrics:
        _print_model_metric_footer("numerics", args)


def add_numerics_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "numerics",
        help="Show a model's accuracy metrics.",
        description="Display per-device numerical accuracy metrics for a model, "
        "alongside the torch reference value. The runtime, precision, chipset, "
        "device, and sdk-version args act as filters.",
    )
    parser.add_argument(
        "model", type=str.lower, help="Model ID or display name (e.g. mobilenet_v2)."
    )
    add_model_metric_filter_args(parser)
    add_version_arg(parser)
    parser.set_defaults(func=_run_numerics)
    return parser


def _dispatch_export_or_evaluate(script: str, raw_args: list[str]) -> None:
    """Run ``qai-hub-models {export,evaluate} <model> [model-args...]``.

    Resolves the model name -> id, then hands the remaining args to the model's
    own ``build_parser()``/``main()`` (via the heavy-side dispatcher) so that
    ``--target-runtime``, ``--device``, etc. are natively parsed instead of
    forwarded through an ``argparse.REMAINDER`` bucket.

    The manifest is always read at ``CURRENT_VERSION``; ``-v/--qaihm-version``
    isn't exposed for these subcommands because the dispatch target is the
    locally-installed model module, not a manifest entry.
    """
    if not raw_args or raw_args[0] in ("-h", "--help"):
        sys.exit(
            f"Usage: {CLI_NAME} {script} <model> [args...]\n"
            f"Run `{CLI_NAME} {script} <model> --help` for the model's options."
        )
    model_arg, forwarded = raw_args[0], list(raw_args[1:])

    from qai_hub_models.utils.path_helpers import MODEL_IDS

    model_id = model_arg
    if model_id not in MODEL_IDS:
        # get_manifest_entry will rebuild the manifest for internal dev versions (which takes a long time).
        # We only call it if model_args does not exactly map to a model ID.
        model_id = get_manifest_entry(model_arg, CURRENT_VERSION).id
        if model_id not in MODEL_IDS:
            sys.exit(
                f"Model {model_id!r} is in the v{CURRENT_VERSION} manifest but not "
                f"in the installed qai_hub_models package. Upgrade with "
                f"`pip install -U qai_hub_models`."
            )

    from qai_hub_models.cli.dispatch import run_model_script

    try:
        run_model_script(model_id=model_id, script=script, forwarded=forwarded)
    except Exception as e:
        raise RuntimeError(
            f"\nSomething went wrong (an exception was thrown). Email us at ai-hub-support@qti.qualcomm.com for assistance.\n\n"
            f"Enable a full stack trace by setting the following environment variable: `{VERBOSE_EXCEPTIONS_ENVVAR}=1`. \n"
            "You should include the stack trace in your support request."
        ) from e


def _warn_missing_qaihm_install(
    _: argparse.Namespace,
    cmd: str,
) -> None:
    install_command = (
        "`pip install -e src -e cli` from the repo root"
        if CURRENT_VERSION.is_devrelease
        else f"`pip install qai-hub-models=={CURRENT_VERSION}`"
    )
    print(
        f"The `qai-hub-models` pypi package is required to run {cmd}, but is not installed (you only have `qai-hub-models-cli` installed).\n"
    )
    print(f"Run {install_command} and try again.")


def add_qaihm_required_help_only_parser(
    subparsers: argparse._SubParsersAction,
    name: str,
    helpmsg: str,
    description: str,
    args: list[tuple[str, Any]],
) -> argparse.ArgumentParser:
    """
    Arg parsing for some commands like `export <model> ...` are
    delegated to the model's own parser in AI Hub Models.

    If AI Hub Models is installed, commands should be intercepted before this parser
    can read them, and forwarded to the appropriate parser defined by AI Hub Models.
    (for example, the args would be forwarded to the parser defined by a model's export script).

    Therefore, this subparser has only two purposes:
        * Makes sure the command shows up in top-level --help message.
        * If a command is not intercepted, AI Hub Models is not installed and this parser
          will read the arguments instead. This parser will always print
          an error message that asks the user to install AI Hub Models.
    """
    parser = subparsers.add_parser(
        name,
        help=helpmsg,
        description=description,
        add_help=False,
    )
    for argname, argtype in args:
        parser.add_argument(argname, type=argtype)
    # This swallows any possible args when AI Hub Models is not installed, so _warn_missing_qaihm_install
    # is called instead of the parser erroring out on unknown arguments.
    parser.add_argument("allow_any_args", nargs=argparse.REMAINDER)
    # This is never actually called by the CLI's main() function unless AI Hub Models is not installed.
    # We "hijack" export calls and send them to the model-specific export / evaluate parser defined in QAIHM.
    parser.set_defaults(func=functools.partial(_warn_missing_qaihm_install, cmd=name))
    return parser


def add_export_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    return add_qaihm_required_help_only_parser(
        subparsers,
        name="export",
        helpmsg="Export a model from source using AI Hub Workbench.",
        description=(
            f"Run `{CLI_NAME} export <model> --help` to see the model's "
            "native export options."
        ),
        args=[("model", str)],
    )


def add_evaluate_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    return add_qaihm_required_help_only_parser(
        subparsers,
        name="evaluate",
        helpmsg="Evaluate a model from source using AI Hub Workbench.",
        description=(
            f"Run `{CLI_NAME} evaluate <model> --help` to see the model's "
            "native evaluate options."
        ),
        args=[("model", str)],
    )


def _run_list_models(args: argparse.Namespace) -> None:
    manifest = get_manifest(args.qaihm_version)

    # All filters except --domain rely on manifest fields added in
    # MIN_MODEL_FILTER_VERSION; older releases only support --domain.
    gated_filters = (
        args.quantized,
        args.runtime,
        args.aot,
        args.jit,
        args.tag,
        args.chipset,
        args.device,
        args.llm,
        args.vlm,
        args.use_case,
    )
    if any(gated_filters) and not feature_supported(
        args.qaihm_version, MIN_MODEL_FILTER_VERSION
    ):
        print(
            f"Filtering by quantization, runtime, chipset, device, tag, or use case "
            f"requires version {MIN_MODEL_FILTER_VERSION} or later. Only --domain is "
            f"supported for version {args.qaihm_version}."
        )
        return

    # Resolve/validate each filter's criteria once up front, then build a list of
    # per-model predicates so the models are walked a single time below.
    predicates: list[Callable[[ManifestModelEntry], bool]] = []

    if args.domain:
        domain_filter = normalize_label(args.domain)
        predicates.append(
            lambda e: normalize_label(domain_proto_to_str(e.domain)) == domain_filter
        )

    if args.use_case:
        use_case_val = use_case_str_to_proto(args.use_case)
        predicates.append(lambda e: e.use_case == use_case_val)

    if args.quantized:
        predicates.append(lambda e: e.is_quantized)

    if args.aot or args.jit or args.runtime:
        platform_runtimes = get_platform(args.qaihm_version).runtimes
        if runtimes := flatten_multi_arg(args.runtime):
            try:
                runtime_vals = {runtime_str_to_proto(r) for r in runtimes}
            except KeyError as e:
                print(str(e))
                return
            predicates.append(lambda e: runtime_vals.issubset(e.supported_runtimes))
        if args.aot:
            aot = {rt.runtime for rt in platform_runtimes if rt.is_aot_compiled}
            predicates.append(lambda e: bool(aot.intersection(e.supported_runtimes)))
        if args.jit:
            jit = {rt.runtime for rt in platform_runtimes if not rt.is_aot_compiled}
            predicates.append(lambda e: bool(jit.intersection(e.supported_runtimes)))

    if tags := flatten_multi_arg(args.tag):
        try:
            tag_vals = {tag_str_to_proto(t) for t in tags}
        except KeyError as e:
            print(str(e))
            return
        predicates.append(lambda e: tag_vals.issubset(e.tags))

    if args.llm:
        predicates.append(lambda e: MODEL_TAG_LLM in e.tags)

    if args.vlm:
        predicates.append(lambda e: MODEL_TAG_VLM in e.tags)

    if args.chipset or args.device:
        try:
            reg = get_platform(args.qaihm_version)
            chipset_name = resolve_chipset(
                reg.chipsets,
                reg.devices,
                chipset=args.chipset,
                device=args.device,
            ).name
        except KeyError as e:
            print(str(e))
            return
        predicates.append(lambda e: chipset_name in e.supported_chipsets)

    entries = sorted(
        (e for e in manifest.models if all(p(e) for p in predicates)),
        key=lambda e: e.id,
    )

    if not entries:
        print("No models found.")
        return

    if args.quiet:
        for entry in entries:
            print(entry.id)
        return

    groups: dict[str, list[ManifestModelEntry]] = {}
    for entry in entries:
        domain = domain_proto_to_str(entry.domain)
        groups.setdefault(domain, []).append(entry)

    # The Use Case/Quantized/Runtimes columns are populated from manifest fields
    # added in MIN_MODEL_FILTER_VERSION; on older releases they'd be blank, so
    # omit them.
    show_filter_columns = feature_supported(
        args.qaihm_version, MIN_MODEL_FILTER_VERSION
    )
    columns = ["Name", "Domain"]
    if show_filter_columns:
        columns += ["Use Case", "Quantized", "Runtimes"]
        wrap_column, wrap_on_commas = "Runtimes", True
    else:
        wrap_column, wrap_on_commas = "Name", False

    platform = get_platform(args.qaihm_version) if show_filter_columns else None
    rows = []
    for domain, group in groups.items():
        for entry in group:
            row = [entry.display_name, domain]
            if show_filter_columns:
                row.append(use_case_proto_to_str(entry.use_case))
                row += [
                    "Yes" if entry.is_quantized else "No",
                    ", ".join(
                        runtime_proto_to_str(
                            r,
                            platform.runtimes if platform else None,
                            display_name=True,
                        )
                        for r in entry.supported_runtimes
                    ),
                ]
            rows.append(row)

    print(
        build_table(
            columns,
            rows,
            wrap_column=wrap_column,
            wrap_on_commas=wrap_on_commas,
            title="Models",
        )
    )

    print(f"Total: {len(entries)} models\n")
    print("Looking for something else?")
    print(
        " - Use AI Hub Workbench to bring your own model: https://aihub.qualcomm.com/get-started#workbench"
    )
    print(
        " - Request we add a new model: https://github.com/qualcomm/ai-hub-models/issues\n"
    )
    print(
        f"More about our supported platforms: `{sample_command('runtimes')}`, "
        f"`{sample_command('devices')}`, `{sample_command('chipsets')}`\n"
    )
    print(
        f"Run `{sample_command('info', '<model_id>')}` for details and download options."
    )
    print_upgrade_notice()


def add_list_models_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "models",
        help="List all available models.",
        description="List all models available in a given AI Hub Models release.",
    )
    add_version_arg(parser)
    domain_values = ", ".join(
        domain_proto_to_str(d)
        for d in ModelDomain.values()
        if d != ModelDomain.MODEL_DOMAIN_UNSPECIFIED
    )
    parser.add_argument(
        "--domain",
        default=None,
        type=str.lower,
        help=f"Filter by domain. Known values: {domain_values}.",
    )
    use_case_values = ", ".join(
        use_case_proto_to_str(u)
        for u in ModelUseCase.values()
        if u != ModelUseCase.MODEL_USE_CASE_UNSPECIFIED
    )
    parser.add_argument(
        "--use-case",
        default=None,
        type=str.lower,
        help=f"Filter by use case. Known values: {use_case_values}.",
    )
    parser.add_argument(
        "--quantized",
        action="store_true",
        help="Filter to quantized models.",
    )
    parser.add_argument(
        "-r",
        "--runtime",
        nargs="+",
        action="append",
        default=None,
        type=str.lower,
        help="Filter to models with assets for all of the given runtimes. "
        "May be repeated or given multiple values. "
        f"Known values: {RUNTIME_VALUES}.",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Filter to Large Language Models that handle text input.",
    )
    parser.add_argument(
        "--vlm",
        action="store_true",
        help="Filter to Vision Language Models that handle text and image input.",
    )
    compile_group = parser.add_mutually_exclusive_group()
    compile_group.add_argument(
        "--aot",
        action="store_true",
        help="Filter to models with ahead-of-time (device-specific) compiled assets.",
    )
    compile_group.add_argument(
        "--jit",
        action="store_true",
        help="Filter to models with just-in-time (universal) compiled assets.",
    )
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "-c",
        "--chipset",
        default=None,
        type=str.lower,
        help="Filter by a chipset the model has been profiled on. "
        f"Run `{sample_command('chipsets')}` to see supported chipsets.",
    )
    target_group.add_argument(
        "-d",
        "--device",
        default=None,
        help="Filter by a device the model has been profiled on. "
        f"Run `{sample_command('devices')}` to see supported devices. "
        "Cannot be combined with --chipset.",
    )
    tag_values = ", ".join(
        tag_proto_to_str(t)
        for t in ModelTag.values()
        if t != ModelTag.MODEL_TAG_UNSPECIFIED
    )
    parser.add_argument(
        "-t",
        "--tag",
        nargs="+",
        action="append",
        default=None,
        type=str.lower,
        help="Filter to models with all of the given tags. "
        "May be repeated or given multiple values. "
        f"Known values: {tag_values}.",
    )
    add_quiet_arg(parser, "Print model IDs only, one per line.")
    parser.set_defaults(func=_run_list_models)
    return parser


def _run_list_devices(args: argparse.Namespace) -> None:
    platform = get_platform(args.qaihm_version)
    devices = sorted(
        platform.devices,
        key=lambda d: (form_factor_proto_to_str(d.form_factor), d.name),
    )
    types = flatten_multi_arg(args.type)
    oses = flatten_multi_arg(args.os)
    devices = filter_devices(
        devices,
        platform.chipsets,
        form_factor=[form_factor_str_to_proto(t) for t in types] if types else None,
        os=[os_str_to_proto(o) for o in oses] if oses else None,
        fp16=True if args.fp16 else None,
        htp_version=flatten_multi_arg(args.htp_version),
        soc_model=flatten_multi_arg(args.soc_model),
    )

    if not devices:
        print("No devices found.")
        return

    if args.quiet:
        for device in devices:
            print(device.name)
        return

    print(format_devices_table(devices, platform.chipsets))
    print(
        f"Total: {len(devices)} devices. This table is a snapshot of devices supported by AI Hub Models v{args.qaihm_version}. AI Hub Workbench supports a different set of devices.\n"
    )
    print(
        f"NOTE: Customized model commands (export, evaluate --- see `{sample_command('--help')}`) support the devices available on AI Hub Workbench rather than this table.\n      To print devices available with AI Hub Workbench, run: `pip install qai_hub; qai-hub list-devices` "
    )

    print(
        f"\nSee all chipsets supported by AI Hub Models using `{sample_command('chipsets', version_flag(args.qaihm_version))}`."
    )

    print_upgrade_notice()


def add_list_devices_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "devices",
        help="List all supported devices.",
        description="List all devices supported in a given AI Hub Models release.",
    )
    add_version_arg(parser)
    type_values = ", ".join(
        form_factor_proto_to_str(f)
        for f in FormFactor.values()
        if f != FormFactor.FORM_FACTOR_UNSPECIFIED
    )
    parser.add_argument(
        "-t",
        "--type",
        nargs="+",
        action="append",
        default=None,
        help=f"Filter by device type(s). Known values: {type_values}.",
    )
    parser.add_argument(
        "--os",
        nargs="+",
        action="append",
        default=None,
        help="Filter by operating system(s) (e.g. Android, Windows).",
    )
    add_chipset_attribute_filter_args(parser)
    add_quiet_arg(parser, "Print device names only, one per line.")
    parser.set_defaults(func=_run_list_devices)
    return parser


def _run_list_chipsets(args: argparse.Namespace) -> None:
    platform = get_platform(args.qaihm_version)
    chipsets = sorted(
        platform.chipsets,
        key=lambda c: (world_proto_to_str(c.world), c.marketing_name),
    )
    types = flatten_multi_arg(args.type)
    chipsets = filter_chipsets(
        chipsets,
        world=[world_str_to_proto(t) for t in types] if types else None,
        fp16=True if args.fp16 else None,
        htp_version=flatten_multi_arg(args.htp_version),
        soc_model=flatten_multi_arg(args.soc_model),
    )

    if not chipsets:
        print("No chipsets found.")
        return

    if args.quiet:
        for chipset in chipsets:
            print(chipset.marketing_name)
        return

    print(format_chipsets_table(chipsets))
    print(
        f"Total: {len(chipsets)} chipsets. This table is a snapshot of chipsets supported by AI Hub Models v{args.qaihm_version}. AI Hub Workbench supports a different set of chipsets.\n"
    )
    print(
        f"NOTE: Customized model commands (export, evaluate --- see `{sample_command('--help')}`) support the chipsets available on AI Hub Workbench rather than this table.\n      To print devices available with AI Hub Workbench, run: `pip install qai_hub; qai-hub list-devices` "
    )

    print(
        f"\nSee all devices supported by AI Hub Models using `{sample_command('devices', version_flag(args.qaihm_version))}`."
    )
    print_upgrade_notice()


def add_list_chipsets_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "chipsets",
        help="List all supported chipsets.",
        description="List all chipsets supported in a given AI Hub Models release.",
    )
    add_version_arg(parser)
    type_values = ", ".join(
        world_proto_to_str(w)
        for w in WebsiteWorld.values()
        if w != WebsiteWorld.WEBSITE_WORLD_UNSPECIFIED
    )
    parser.add_argument(
        "-t",
        "--type",
        nargs="+",
        action="append",
        default=None,
        help=f"Filter by chipset type(s). Known values: {type_values}.",
    )
    add_chipset_attribute_filter_args(parser)
    add_quiet_arg(parser, "Print chipset names only, one per line.")
    parser.set_defaults(func=_run_list_chipsets)
    return parser


def _run_list_runtimes(args: argparse.Namespace) -> None:
    runtimes = get_platform(args.qaihm_version).runtimes

    if args.quiet:
        for rt in runtimes:
            print(runtime_proto_to_str(rt.runtime))
        return

    print(format_runtimes_table(runtimes, args.qaihm_version))
    print(f"Total: {len(runtimes)} runtimes")
    # Display metadata (incl. docs links) exists only as of MIN_MODEL_FILTER_VERSION.
    if feature_supported(args.qaihm_version, MIN_MODEL_FILTER_VERSION) and (
        links := format_runtime_links(runtimes)
    ):
        print(f"\n{links}")
    print_upgrade_notice()


def add_list_runtimes_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "runtimes",
        help="List all runtimes.",
        description="List all runtimes models can be compiled to.",
    )
    add_version_arg(parser)
    add_quiet_arg(parser, "Print runtime IDs only, one per line.")
    parser.set_defaults(func=_run_list_runtimes)
    return parser


def _run_info(args: argparse.Namespace) -> None:
    info = get_model_info(args.model, args.qaihm_version)

    url = f"{AIHUB_MODELS_URL}/{info.id}"
    width = max(len(info.name), len(url)) + 4
    print("+" + "=" * width + "+")
    print(f"| {info.name:^{width - 2}} |")
    print(f"| {url:^{width - 2}} |")
    print("+" + "=" * width + "+")
    print()

    if info.description:
        desc_table = PrettyTable()
        desc_table.title = "Description"
        desc_table.header = False
        desc_table.align = "l"
        desc_table.add_row([info.description])
        wrap_table_column(desc_table, 0)
        print(desc_table)
        print()

    metadata_table = PrettyTable()
    metadata_table.header = False
    metadata_table.align = "l"
    if info.domain:
        metadata_table.add_row(["Domain", domain_proto_to_str(info.domain)])
    if info.use_case:
        metadata_table.add_row(["Use Case", use_case_proto_to_str(info.use_case)])
    if info.tags:
        metadata_table.add_row(
            ["Tags", ", ".join(tag_proto_to_str(t) for t in info.tags)]
        )
    # is_quantized / supported_runtimes are manifest fields added in 0.56.0.
    # Best-effort: skip if the model isn't in the manifest for this version.
    if feature_supported(args.qaihm_version, MIN_MODEL_FILTER_VERSION):
        try:
            entry = get_manifest_entry(args.model, args.qaihm_version)
        except KeyError:
            entry = None
        if entry is not None:
            metadata_table.add_row(["Quantized", "Yes" if entry.is_quantized else "No"])
            if entry.supported_runtimes:
                metadata_platform = get_platform(args.qaihm_version)
                metadata_table.add_row(
                    [
                        "Supported Runtimes",
                        ", ".join(
                            runtime_proto_to_str(
                                r, metadata_platform.runtimes, display_name=True
                            )
                            for r in entry.supported_runtimes
                        ),
                    ]
                )
            if entry.supported_chipsets:
                chipset_names = {
                    c.name: c.marketing_name
                    for c in get_platform(args.qaihm_version).chipsets
                }
                metadata_table.add_row(
                    [
                        "Supported Chipsets",
                        ", ".join(
                            chipset_names.get(c, c) for c in entry.supported_chipsets
                        ),
                    ]
                )
    if info.license_type:
        license_str = license_proto_to_str(info.license_type)
        if info.HasField("license_url"):
            license_str += f" ({info.license_url})"
        metadata_table.add_row(["License", license_str])
    if info.HasField("source_repo"):
        metadata_table.add_row(["Source Repo", info.source_repo])
    if info.HasField("research_paper"):
        title = (
            info.research_paper_title
            if info.HasField("research_paper_title")
            else "Paper"
        )
        metadata_table.add_row(["Paper", f"{title} ({info.research_paper})"])
    if metadata_table.rows:
        metadata_table.title = "Metadata"
        wrap_table_column(metadata_table, 1)
        print(metadata_table)
        print()

    def _technical_details_table(title: str, details: Iterable) -> PrettyTable:
        table = PrettyTable()
        table.title = title
        table.header = False
        table.align = "l"
        for detail in details:
            if detail.HasField("string_value"):
                val = detail.string_value
            elif detail.HasField("int_value"):
                val = str(detail.int_value)
            elif detail.HasField("float_value"):
                val = str(detail.float_value)
            else:
                val = ""
            table.add_row([detail.key, val])
        wrap_table_column(table, 1)
        return table

    if info.technical_details:
        print(_technical_details_table("Technical Details", info.technical_details))
        print()

    for rt_details in info.runtime_technical_details:
        runtime_name = runtime_proto_to_str(
            rt_details.runtime,
            get_platform(args.qaihm_version).runtimes,
            display_name=True,
        )
        print(
            _technical_details_table(
                f"Technical Details ({runtime_name})",
                rt_details.technical_details,
            )
        )
        print()

    try:
        release_assets = get_model_release_assets(args.model, args.qaihm_version)
        info_platform = get_platform(args.qaihm_version)
        print(
            format_release_assets_table(
                release_assets,
                info_platform.chipsets,
                title="Download Options",
                runtimes=info_platform.runtimes,
            )
        )
        print()
        print(
            format_fetch_commands(
                release_assets,
                args.model,
                include_metrics=True,
                version=args.qaihm_version,
            )
        )
        print()
        print(
            f"Most models can be further customized beyond what is offered by standard model downloads. Scripts that can export the model from source are available at {model_repo_url(info.id, args.qaihm_version)}"
        )
    except (FileNotFoundError, UnsupportedVersionError) as e:
        err_table = PrettyTable()
        err_table.title = "Download Options"
        err_table.header = False
        err_table.align = "l"
        err_table.add_row([str(e)])
        wrap_table_column(err_table, 0)
        print(err_table)


def add_info_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "info",
        help="Show detailed information about a model.",
        description="Display model metadata including description, license, "
        "technical details, and available download options.",
    )
    parser.add_argument(
        "model",
        type=str.lower,
        help="Model ID or display name (e.g. mobilenet_v2).",
    )
    add_version_arg(parser)
    parser.set_defaults(func=_run_info)
    return parser


def _run_versions(args: argparse.Namespace) -> None:
    supported = get_supported_versions(force_refresh=True)
    installed = CURRENT_VERSION

    if args.quiet:
        print(", ".join(str(v) for v in supported))
        return

    print("Supported AI Hub Models Versions:")
    labeled = [f"{v} (installed)" if v == installed else str(v) for v in supported]
    print(", ".join(labeled))
    print_upgrade_notice()


def add_versions_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "versions",
        help="List all AI Hub Models versions supported by this CLI.",
        description="List all AI Hub Models versions supported by this CLI. "
        "Shows which version is currently installed and whether newer versions are available.",
    )
    add_quiet_arg(
        parser,
        "Print versions as a plain comma-separated list without the (installed) marker or upgrade notice.",
    )
    parser.set_defaults(func=_run_versions)
    return parser


def _run_validate_aws(args: argparse.Namespace) -> None:
    from qai_hub_models_cli._internal.aws import validate_credentials

    validate_credentials()


def add_validate_aws_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "validate_aws_credentials",
        help="Validate and refresh AWS credentials for internal release access.",
        description="Ensure the 'qaihm' AWS profile has valid credentials. "
        "If credentials are expired, refreshes them via saml2aws. "
        "Requires the [internal] extra (pip install qai_hub_models_cli[internal]).",
    )
    parser.set_defaults(func=_run_validate_aws)
    return parser


class _GroupedHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Lists subcommands grouped into titled sections in the top-level help.

    argparse has no native support for sectioning subcommands; it renders every
    choice in one flat ``positional arguments`` block. This formatter suppresses
    that block and appends a grouped rendering driven by ``self.sections``: an
    ordered ``{title: [command_name, ...]}`` dict. The subparsers action and the
    sections are passed in at construction (via ``functools.partial`` as the
    ``formatter_class``), so neither method depends on the other's call order.
    Commands absent from the action (e.g. the conditionally-registered
    ``validate_aws_credentials``) are skipped, so the same mapping works
    regardless of which subcommands are present.
    """

    def __init__(
        self,
        *args: Any,
        subparsers_action: argparse.Action | None = None,
        sections: dict[str, list[str]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._subparsers_action = subparsers_action
        self.sections = sections or {}

    def _format_action(self, action: argparse.Action) -> str:
        if action is self._subparsers_action:
            return ""
        return super()._format_action(action)

    def format_help(self) -> str:
        text = super().format_help()
        action = self._subparsers_action
        if action is None or not self.sections:
            return text

        registered: dict[str, argparse.ArgumentParser] = dict(action.choices or {})
        help_by_name = {
            a.dest: (a.help or "") for a in getattr(action, "_choices_actions", [])
        }

        def invocation(name: str) -> str:
            """``name`` followed by its subparser's positional metavars.

            Each positional is rendered as ``<dest>`` (with nargs decoration,
            e.g. ``[<model>]`` for an optional one) to match the ``<model>``
            placeholder convention used elsewhere in the CLI's help text.
            """
            subparser = registered[name]
            positionals = [
                self._format_args(a, f"<{a.dest}>")
                for a in subparser._get_positional_actions()
            ]
            return " ".join([name, *positionals])

        displayed = [
            (title, [(n, invocation(n)) for n in names if n in registered])
            for title, names in self.sections.items()
        ]
        width = max(
            (len(label) for _, items in displayed for _, label in items), default=0
        )
        blocks = []
        for title, items in displayed:
            if not items:
                continue
            lines = [f"{title}:"]
            lines += [
                f"  {label:<{width}}  {help_by_name.get(name, '')}"
                for name, label in items
            ]
            blocks.append("\n".join(lines))

        blocks.append(
            "Many commands accept flags to narrow the results.\n"
            "Run 'qai-hub-models <command> --help' to "
            "see the flags a command supports."
        )
        return text.rstrip() + "\n\n" + "\n\n".join(blocks) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser with every subcommand registered.

    Extracted from ``main`` so tests can introspect which subcommands are
    available (e.g. that ``export``/``evaluate`` only show up when the heavy
    ``qai_hub_models`` package is installed) without spawning a process or
    capturing stdout.
    """
    parser = argparse.ArgumentParser(
        prog=CLI_NAME,
        description="Qualcomm AI Hub Models CLI.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {CURRENT_VERSION}",
    )
    subparsers = parser.add_subparsers(metavar="<command>", dest="command")

    add_fetch_parser(subparsers)
    add_info_parser(subparsers)
    add_perf_parser(subparsers)
    add_numerics_parser(subparsers)
    add_list_models_parser(subparsers)
    add_list_devices_parser(subparsers)
    add_list_chipsets_parser(subparsers)
    add_list_runtimes_parser(subparsers)
    add_find_parser(subparsers)
    add_versions_parser(subparsers)
    add_export_parser(subparsers)
    add_evaluate_parser(subparsers)
    if use_internal_releases() or is_internal_repo():
        add_validate_aws_parser(subparsers)

    sections = {
        "Models": [
            "fetch",
            "info",
            "perf",
            "numerics",
            "find",
        ],
        "Customized Models (export from source)": [
            "export",
            "evaluate",
        ],
        "Catalog": [
            "models",
            "devices",
            "chipsets",
            "runtimes",
            "versions",
        ],
        "Qualcomm Internal": ["validate_aws_credentials"],
    }
    parser.formatter_class = partial(
        _GroupedHelpFormatter,
        subparsers_action=subparsers,
        sections=sections,
    )

    return parser


def main(args: list[str] | None = None) -> None:
    _check_version_match()

    # `export`/`evaluate` route through the model's own parser, not ours. Sniff
    # for them before constructing the lean parser so argparse doesn't try to
    # consume `--help` or unknown model-specific flags at this layer.
    raw = sys.argv[1:] if args is None else args
    if raw and raw[0] in ("export", "evaluate") and is_heavy_package_installed():
        try:
            _dispatch_export_or_evaluate(raw[0], raw[1:])
        except Exception as e:
            if bool_envvar_value(VERBOSE_EXCEPTIONS_ENVVAR):
                raise
            print(e.args[0] if isinstance(e, KeyError) and e.args else e)
            sys.exit(1)
        return

    parser = _build_parser()
    # parse_known_args lets us route an unrecognized-arg error to the chosen
    # subcommand's parser (showing its usage) instead of the top-level one.
    parsed, extras = parser.parse_known_args(args)
    if extras:
        subparsers_action = next(
            a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
        )
        cmd = getattr(parsed, "command", None)
        target = subparsers_action.choices.get(cmd, parser) if cmd else parser
        target.error(f"unrecognized arguments: {' '.join(extras)}")
    if hasattr(parsed, "func"):
        try:
            parsed.func(parsed)
        except Exception as e:
            if bool_envvar_value(VERBOSE_EXCEPTIONS_ENVVAR):
                raise
            # KeyError.__str__ wraps its message in quotes (it uses repr); print
            # the message text directly so error output reads cleanly.
            print(e.args[0] if isinstance(e, KeyError) and e.args else e)
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
