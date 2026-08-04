# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

"""Utility Functions for parsing input args for export and other customer facing scripts."""

from __future__ import annotations

import argparse
import inspect
import sys
import warnings
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from functools import partial
from itertools import chain
from pathlib import Path
from pydoc import locate
from typing import Any, TypeVar, cast

import qai_hub as hub
from numpydoc.docscrape import FunctionDoc

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.protocols import (
    FromPretrainedProtocol,
    FromPretrainedTypeVar,
)
from qai_hub_models.utils.ai_hub_access import can_access_qualcomm_ai_hub
from qai_hub_models.utils.base_collection_model import (
    CollectionModel,
    WorkbenchModelCollection,
)
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_model import (
    PrecompiledWorkbenchModel,
    WorkbenchModel,
)
from qai_hub_models.utils.device import resolve_hub_device
from qai_hub_models.utils.envvars import DevModeEnvvar
from qai_hub_models.utils.evaluate.helpers import EvalMode
from qai_hub_models.utils.inference import OnDeviceModel, compile_model_from_args
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.kwarg_helpers import filter_kwargs, get_params
from qai_hub_models.utils.qai_hub_helpers import (
    raise_if_fp_is_unsupported,
)


class ParseEnumAction(argparse.Action):
    def __init__(
        self,
        option_strings: list[str],
        dest: str,
        enum_type: type[Enum],
        **kwargs: Any,
    ) -> None:
        super().__init__(option_strings, dest, **kwargs)
        self.enum_type = enum_type

    def __call__(  # type: ignore[override]
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | None,
        option_string: str | None = None,
    ) -> None:
        assert isinstance(values, str)
        setattr(namespace, self.dest, self.enum_type[values.upper().replace("-", "_")])


ParserT = TypeVar("ParserT", bound=argparse.ArgumentParser)


def _get_non_float_precision(
    supported_precisions: set[Precision] | None,
) -> Precision | None:
    if not supported_precisions:
        return None

    for p in supported_precisions:
        if p != Precision.float:
            return p

    return None


def get_quantize_action_with_default(
    default_quantized_precision: Precision,
) -> type[argparse.Action]:
    """
    Get an action that:

    Returns default_quantized_precision if "--quantize" is passed with no arg.

    Returns a parsed precision object if "--quantize <value> " is passed.
    """

    class ParsePrecisionAction(argparse.Action):
        def __init__(self, option_strings: list[str], dest: str, **kwargs: Any) -> None:
            super().__init__(option_strings, dest, **kwargs)

        def __call__(  # type: ignore[override]
            self,
            parser: argparse.ArgumentParser,
            namespace: argparse.Namespace,
            values: str | Precision | None,
            option_string: str | None = None,
        ) -> None:
            if values:
                if isinstance(values, Precision):
                    val = values
                else:
                    assert isinstance(values, str)
                    val = Precision.parse(values)
            else:
                val = default_quantized_precision

            setattr(namespace, self.dest, val)

    return ParsePrecisionAction


class QAIHMHelpFormatter(
    argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    """
    Argparse formatter that combnines:
      * allowing raw text (eg. newlines) in help messages
      * including defaults in help messages (except for boolean args)
    """

    def _get_help_string(self, action: argparse.Action) -> str | None:
        """
        Default value for booleans in CLI help can be misleading.
        This overridden function will print just the help message for boolean args
        and print help message along with the default value for all other args.
        """
        # Don't print "(default: <value>)" in the CLI help if the value is a bool
        # or something "non-truthy" (e.g. "", None, [])
        if isinstance(
            action, (argparse._StoreTrueAction, argparse._StoreFalseAction)
        ) or (not action.default):
            return action.help
        return super()._get_help_string(action)


class QAIHMArgumentParser(argparse.ArgumentParser):
    """
    An ArgumentParser that sets device from the appropriate options.
    This isn't implemented as a `type` argument to `add_argument` because the
    device/chipset can be modified by device_os.
    """

    def __init__(
        self,
        model_cls: type[FromPretrainedTypeVar] | None = None,
        supported_precision_runtimes: (
            dict[Precision, list[TargetRuntime]] | None
        ) = None,
        default_device: str | None = None,
        default_chipset: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.supported_precision_runtimes = supported_precision_runtimes or {}
        self.default_device = default_device
        self.default_chipset = default_chipset
        self.model_cls = model_cls
        self._dataset_name_to_cls: dict[str, type[BaseDataset]] = {}
        super().__init__(*args, **kwargs)

    def set_supported_dataset_classes(
        self, dataset_classes: Sequence[type[BaseDataset]]
    ) -> None:
        self._dataset_name_to_cls = {
            ds_cls.dataset_name(): ds_cls for ds_cls in dataset_classes
        }

    @staticmethod
    def get_hub_device(
        device: str | None = None, chipset: str | None = None, device_os: str = ""
    ) -> hub.Device | None:
        """
        Get a fully-resolved hub.Device given a device name and/or chipset.
        If neither is specified, returns None.
        """
        if not device and not chipset:
            return None
        return resolve_hub_device(
            name=device or "", chipset=chipset or "", os=device_os
        )

    def parse_args(  # type: ignore[override]
        self, args: list[str] | None = None, namespace: argparse.Namespace | None = None
    ) -> argparse.Namespace:
        parsed = super().parse_args(args, namespace or argparse.Namespace())
        parsed.device = self.get_hub_device(
            getattr(parsed, "device_str", None),
            getattr(parsed, "chipset", None),
            getattr(parsed, "device_os", ""),
        )
        if parsed.device is None:
            parsed.device = self.get_hub_device(
                self.default_device, self.default_chipset
            )

        if getattr(parsed, "quantize", None):
            parsed.precision = parsed.quantize

        # Resolve default target_runtime based on the chosen precision.
        if getattr(parsed, "target_runtime", None) is None:
            precision = getattr(parsed, "precision", None)
            if precision is None and self.supported_precision_runtimes:
                precision = next(iter(self.supported_precision_runtimes))
            if precision is not None and precision in self.supported_precision_runtimes:
                parsed.target_runtime = _get_default_runtime(
                    self.supported_precision_runtimes[precision]
                )
            else:
                # No precision arg -- fall back to first eligible across all precisions.
                all_runtimes = chain.from_iterable(
                    self.supported_precision_runtimes.values()
                )

                # If all else fails, use TFLITE
                parsed.target_runtime = next(all_runtimes, TargetRuntime.TFLITE)

        if getattr(parsed, "target_runtime", None) == TargetRuntime.GENIE:
            warnings.warn(
                "--target-runtime genie is deprecated and will be removed in a "
                "future release. Please migrate to --target-runtime geniex_qairt.",
                DeprecationWarning,
                stacklevel=2,
            )

        quantized_model_id_arg = getattr(parsed, "quantized_model_id", None)
        if quantized_model_id_arg:
            if getattr(parsed, "precision", None) == Precision.float:
                raise ValueError(
                    "--quantized-model-id can only be used with a quantized precision. "
                    "Pass --precision <non-float> or --quantize."
                )
            assert self.model_cls is not None
            if not issubclass(self.model_cls, CollectionModel):
                # WorkbenchModel
                parsed.quantized_model_id = quantized_model_id_arg
            else:
                # CollectionModel
                components = getattr(parsed, "components", None)
                model_ids = [
                    s.strip() for s in quantized_model_id_arg.split(",") if s.strip()
                ]
                parsed.quantized_model_id = dict(
                    zip(components or [], model_ids, strict=True)
                )

        if self.supported_precision_runtimes:
            self.validate_precision_runtime(self.supported_precision_runtimes, parsed)

        # FP16 device-precision validation
        precision = getattr(parsed, "precision", None)
        if parsed.device is not None and precision is not None:
            self._validate_fp16_support(parsed.device, precision)

        if self._dataset_name_to_cls and hasattr(parsed, "dataset_name"):
            parsed.dataset_cls = self._dataset_name_to_cls[parsed.dataset_name]

        return parsed

    @staticmethod
    def _validate_fp16_support(device: hub.Device, precision: Precision) -> None:
        """Check FP16 support using YAML first, falling back to workbench API."""
        raise_if_fp_is_unsupported(device, precision)

    @staticmethod
    def validate_precision_runtime(
        supported_precision_runtimes: dict[Precision, list[TargetRuntime]],
        parsed_args: argparse.Namespace,
    ) -> None:
        """
        Verifies that supported_precision_runtimes contains the precision + runtime pair chosen by the parsed argument namespace.
        If the namespace does not include both precision and runtime, then validation is skipped.
        """
        # If precision or target_runtime are None, they aren't args used by this parser. This validation becomes a no-op.
        precision: Precision | None = getattr(parsed_args, "precision", None)
        target_runtime: TargetRuntime | None = getattr(
            parsed_args, "target_runtime", None
        )

        if precision is None or target_runtime is None or DevModeEnvvar.get():
            return

        if (
            precision not in supported_precision_runtimes
            or target_runtime not in supported_precision_runtimes[precision]
        ):
            str_supported_precision_runtimes = "\n".join(
                f"    {p}: {', '.join([rt.value for rt in rts])}"
                for p, rts in supported_precision_runtimes.items()
            )
            print(
                f"Model does not support runtime {target_runtime.value} with precision {precision}. These combinations are supported:\n"
                + str_supported_precision_runtimes
            )
            sys.exit(1)


def get_parser(
    model_cls: type[FromPretrainedTypeVar] | None = None,
    supported_precision_runtimes: dict[Precision, list[TargetRuntime]] | None = None,
    allow_dupe_args: bool = False,
) -> QAIHMArgumentParser:
    return QAIHMArgumentParser(
        model_cls,
        supported_precision_runtimes=supported_precision_runtimes,
        formatter_class=QAIHMHelpFormatter,
        conflict_handler="resolve" if allow_dupe_args else "error",
    )


def _add_device_args(
    parser: QAIHMArgumentParser,
    default_device: str | None = None,
    default_chipset: str | None = None,
    required: bool = False,
) -> QAIHMArgumentParser:
    # This is an assertion because this is a logic error; it shouldn't be possible to get this at runtime.
    assert not (default_device and default_chipset), (
        "Only one of default_device or default_chipset may be specified."
    )

    parser.default_device = default_device
    parser.default_chipset = default_chipset
    device_group = parser.add_argument_group("Device Selection")
    device_mutex_group = device_group.add_mutually_exclusive_group(required=required)
    device_mutex_group.add_argument(
        "--device",
        "-d",
        dest="device_str",
        type=str,
        help="The name of the device used to run this script. Run `qai-hub list-devices` to see the list of options."
        + (f" If not set, defaults to `{default_device}`." if default_device else ""),
    )
    device_mutex_group.add_argument(
        "--chipset",
        "-c",
        type=str,
        help="If set, will choose a random device with this chipset. Run `qai-hub list-devices` to see the list of options."
        + (f" If not set, defaults to `{default_chipset}`." if default_chipset else ""),
    )
    device_group.add_argument(
        "--device-os",
        type=str,
        default="",
        help="Optionally specified together with --device or --chipset",
    )

    return parser


def add_output_dir_arg(parser: ParserT) -> ParserT:
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="If specified, saves demo output (e.g. image) to this directory instead of displaying.",
    )
    return parser


def _get_default_runtime(
    available_runtimes: list[TargetRuntime],
) -> TargetRuntime:
    if len(available_runtimes) == 0:
        raise RuntimeError("available_runtimes empty, expecting at-least one runtime.")
    return available_runtimes[0]


def add_target_runtime_arg(
    parser: ParserT,
    helpmsg: str,
    available_target_runtimes: list[TargetRuntime] | None = None,
    default: TargetRuntime | None = None,
    required: bool = False,
) -> ParserT:
    if available_target_runtimes is None:
        available_target_runtimes = list(TargetRuntime.__members__.values())
    parser.add_argument(
        "--runtime",
        "--target-runtime",
        "-r",
        type=str,
        dest="target_runtime",
        action=partial(ParseEnumAction, enum_type=TargetRuntime),  # type: ignore[arg-type]
        default=default,
        required=required,
        metavar=f"{{{', '.join(rt.value for rt in available_target_runtimes)}}}",
        help=helpmsg,
    )
    return parser


def add_precision_arg(
    parser: argparse.ArgumentParser,
    supported_precisions: set[Precision],
    default_if_arg_explicitly_passed: Precision,  # the default value if --precision is passed explicitly
    default: Precision,  # the default value if --precision is not passed
    required: bool = False,
) -> argparse.ArgumentParser:
    precision_help = "Desired precision to which the model should be quantized."
    if Precision.float in supported_precisions:
        precision_help += " If set to 'float', the model will not be quantized, and inference will run in fp32 or fp16 (depending on compute unit)."

    group = parser.add_mutually_exclusive_group(required=required)
    group.add_argument(
        "--precision",
        "-p",
        action=get_quantize_action_with_default(default),
        default=default,
        metavar=f"{{{', '.join(str(p) for p in supported_precisions)}}}",
        help=precision_help,
    )
    if len(supported_precisions) > 1:
        quantize_options = [
            str(p) for p in supported_precisions if p != Precision.float
        ]
        group.add_argument(
            "--quantize",
            action=get_quantize_action_with_default(default_if_arg_explicitly_passed),
            default=None,
            metavar=f"{{{', '.join(quantize_options)}}}",
            help=f"Quantize the model to this precision. If passed without an explicit argument, precision {default_if_arg_explicitly_passed} will be used. If set, this always supercedes the '--precision' argument.",
            nargs="?",
        )
    return parser


def get_on_device_demo_parser(
    parser: QAIHMArgumentParser | None = None,
    supported_eval_modes: list[EvalMode] | None = None,
    supported_precisions: set[Precision] | None = None,
    available_target_runtimes: list[TargetRuntime] | None = None,
    add_output_dir: bool = False,
    default_device: str | None = None,
) -> QAIHMArgumentParser:
    """
    Get argument parser for on-device demo scripts.

    Parameters
    ----------
    parser
        Existing parser to add arguments to. If None, creates a new parser.
    supported_eval_modes
        Subset of EvalMode.{FP,QUANTSIM,ON_DEVICE,LOCAL_DEVICE}. Default is
        [EvalMode.FP, EvalMode.ON_DEVICE]. The first value of supported_eval_modes
        will be the default.
    supported_precisions
        Subset of {Precision.float, Precision.w8a8, Precision.w8a16}
    available_target_runtimes
        Available target runtimes for this model.
    add_output_dir
        Whether to add an output directory argument.
    default_device
        Default device to use for on-device execution.

    Returns
    -------
    parser : QAIHMArgumentParser
        Argument parser with all required arguments for on-device demos.
    """
    if available_target_runtimes is None:
        available_target_runtimes = list(TargetRuntime.__members__.values())
    if not parser:
        parser = get_parser()

    # Add --eval-mode
    supported_eval_modes = supported_eval_modes or [EvalMode.FP, EvalMode.ON_DEVICE]

    # take the first allowed mode as the default
    default_mode = supported_eval_modes[0]

    mode_help_lines = ["Run the model in one of the following modes:"]
    mode_help_lines.extend(
        f"  - {m.value}: {m.description}" for m in supported_eval_modes
    )
    mode_help_msg = "\n".join(mode_help_lines)

    parser.add_argument(
        "--eval-mode",
        type=EvalMode.from_string,
        choices=supported_eval_modes,
        default=default_mode,
        help=mode_help_msg,
    )

    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="If mode==on-device, uses this model Hub model ID."
        " Provide comma separated model-ids if multiple models are required for demo."
        " Run export.py to get on-device demo command with models exported for you.",
    )

    _add_device_args(parser, default_device=default_device)

    if add_output_dir:
        add_output_dir_arg(parser)
    parser.add_argument(
        "--inference-options",
        type=str,
        default="",
        help="If running on-device, use these options when submitting the inference job.",
    )
    default_runtime = _get_default_runtime(available_runtimes=available_target_runtimes)
    add_target_runtime_arg(
        parser,
        helpmsg="The runtime to demo (if `--eval-mode on-device` is specified).",
        default=default_runtime,
        available_target_runtimes=available_target_runtimes,
    )

    # TODO: This should only include supported precisions.
    default_precisions = [Precision.float, Precision.w8a8, Precision.w8a16]
    new_supported_precisions = supported_precisions or default_precisions

    add_precision_arg(
        parser,
        set(new_supported_precisions),
        next(iter(new_supported_precisions)),
        next(iter(new_supported_precisions)),
    )

    return parser


def validate_on_device_demo_args(args: argparse.Namespace, model_name: str) -> None:
    """
    Validates the the args for the on device demo are valid.

    Intended for use only in CLI scripts.
    Prints error to console and exits if an error is found.
    """
    is_on_device = args.eval_mode == EvalMode.ON_DEVICE
    if is_on_device and not can_access_qualcomm_ai_hub():
        raise ValueError(
            "On-device demos (--eval-mode on-device) are not available without Qualcomm® AI Hub Workbench access.\n",
            "Please sign up for Qualcomm® AI Hub Workbench at https://myaccount.qualcomm.com/signup .",
        )

    if is_on_device and (args.device is None):
        raise ValueError(
            "--device or --chipset must be specified with --eval-mode on-device."
        )

    if (args.inference_options or args.hub_model_id) and not is_on_device:
        raise ValueError(
            "A Hub model ID and inference options can be provided only with --eval-mode on-device."
        )


def add_function_parser_args(
    signature: dict[str, inspect.Parameter],
    parser: QAIHMArgumentParser,
    help_fn: Callable[[str, Any], str],
) -> None:
    """
    Given a function signature, add the inputs to the function as args to the parser.

    Parameters
    ----------
    signature
        The function signature represented as
        a dict from arg name to parameter metadata.
    parser
        The parser object to which the args are added.
    help_fn
        A function that takes the argument name and the default value
        and returns the help string for the that arg.
    """
    for name, param in signature.items():
        # Skip variadic parameters (*args / **kwargs). These have no fixed name
        # and cannot be expressed as CLI arguments; a generic signature such as
        # `from_pretrained(*args, **kwargs)` would otherwise produce bogus
        # `--args` / `--kwargs` flags.
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        # Determining type from param.annotation is non-trivial (it can be a
        # strings like "bool | None").
        bool_action = None
        arg_name = f"--{name.replace('_', '-')}"
        type_: Any
        if param.annotation == "list[int]":
            type_ = _parse_int_list
        elif param.default is not None:
            type_ = type(param.default)
            # resolved_type = _resolve_param_type(param, model_cls)
            if type_ is bool:
                if param.default:
                    bool_action = "store_false"
                    # If the default is true, and the arg name does not start with no_,
                    # then add the no- to the argument (as it should be passed as --no-enable-flag, not --enable-flag)
                    if name.startswith("no_"):
                        arg_name = f"--{name[3:].replace('_', '-')}"
                    elif name.startswith("skip_"):
                        arg_name = f"--do-{name[5:].replace('_', '-')}"
                    else:
                        arg_name = f"--no-{name.replace('_', '-')}"
                else:
                    bool_action = "store_true"
                    # If the default is false, and the arg name starts with no_,
                    # then remove the no- from the argument (as it should be passed as --enable-flag, not --no-enable-flag)
                    arg_name = f"--{name.replace('_', '-')}"
        elif param.annotation == "bool":
            type_ = bool
        elif param.annotation == "int | None":
            # Default is None so the type can't be inferred from it; the
            # annotation says it's an int, so parse it as int rather than str.
            type_ = int
        else:
            type_ = str

        help_str = help_fn(name, param.default)
        if bool_action:
            parser.add_argument(arg_name, dest=name, action=bool_action, help=help_str)
        elif isinstance(type_, type) and issubclass(type_, Enum):
            parser.add_argument(
                arg_name,
                type=str,
                action=partial(ParseEnumAction, enum_type=type_),  # type: ignore[arg-type]
                default=param.default,
                choices=[
                    enum.name.lower() for enum in list(type_.__members__.values())
                ],
                help=help_str,
            )
        else:
            parser.add_argument(
                arg_name,
                dest=name,
                type=type_,
                default=param.default,
                help=help_str,
            )


def get_model_cli_parser(
    cls: type[FromPretrainedTypeVar],
    parser: QAIHMArgumentParser | None = None,
    suppress_help_arguments: list | None = None,
    allow_dupe_args: bool = True,
) -> QAIHMArgumentParser:
    """
    Generate the argument parser to create this model from an argparse namespace.
    Default behavior is to assume the CLI args have the same names as from_pretrained method args.
    """
    if not parser:
        parser = get_parser(allow_dupe_args=allow_dupe_args)

    from_pretrained_sig = inspect.signature(cls.from_pretrained)

    export_docs = {
        param.name: "\n".join(param.desc)
        for param in FunctionDoc(cls.from_pretrained)["Parameters"]
    }

    def get_help(name: str, default_value: Any) -> str:
        # Suppress help for argument that need not be exposed for model.
        arg_name = f"--{name.replace('_', '-')}"
        if suppress_help_arguments is not None and arg_name in suppress_help_arguments:
            return argparse.SUPPRESS

        helpmsg = export_docs.get(
            name,
            (
                f"For documentation, see {cls.__name__}::from_pretrained::parameter {name}."
            ),
        )
        if default_value is True:
            helpmsg = f"{helpmsg} Setting this flag will set parameter {name} to False."
        elif default_value is False:
            helpmsg = f"{helpmsg} Setting this flag will set parameter {name} to True."
        return helpmsg

    signature = dict(from_pretrained_sig.parameters)
    if "cls" in signature:
        signature.pop("cls")
    if "precision" in signature:
        signature.pop("precision")
    add_function_parser_args(signature, parser, get_help)
    return parser


def add_input_spec_args(
    cls: type[WorkbenchModel | CollectionModel],
    parser: QAIHMArgumentParser,
) -> QAIHMArgumentParser:
    """Adds arguments from get_input_spec."""
    return get_model_input_spec_parser(cls, parser)


def get_model_kwargs(
    model_cls: type[FromPretrainedTypeVar], args_dict: Mapping[str, Any]
) -> Mapping[str, Any]:
    """
    Given a dict with many args, pull out the ones relevant
    to constructing the model via `from_pretrained`.
    """
    from_pretrained_sig = inspect.signature(model_cls.from_pretrained)
    model_kwargs = {}
    for name in from_pretrained_sig.parameters:
        if name in ["cls", "kwargs"] or name not in args_dict:
            continue
        model_kwargs[name] = args_dict.get(name)
    return model_kwargs


def get_export_model_name(
    model_cls: type[FromPretrainedTypeVar],
    model_id: str,
    precision: Precision | None,
    model_kwargs: Mapping[str, Any],
) -> str:
    """
    When exporting a model with custom model_kwargs, use a different name
    for the model file saved to disk. Incorporate all customized string args
    into the name.
    """
    sig = inspect.signature(model_cls.from_pretrained, eval_str=True)

    name = model_id
    if precision is not None:
        name += f"_{precision}"
    for key, value in sig.parameters.items():
        # Check for a simple string type.
        anno = value.annotation

        if key not in model_kwargs:
            continue

        from types import UnionType

        if not (
            anno is str
            or (
                isinstance(anno, UnionType) and any(arg is str for arg in anno.__args__)
            )
        ):
            continue

        if model_kwargs[key] != sig.parameters[key].default:
            # If the weights are a url or filepath, .stem will take the final name
            # in the path, it will also trim the suffix (i.e., yolov8n.pt -> yolov8n)
            # Note: if a string arg has a '/' character that is not part of a path,
            # we will erroneously truncate everything before it, which we're ok with.
            name += f"_{Path(str(model_kwargs[key])).stem}"
    return name


def model_from_cli_args(
    model_cls: type[FromPretrainedTypeVar], cli_args: argparse.Namespace
) -> FromPretrainedTypeVar:
    """
    Create this model from an argparse namespace.
    Default behavior is to assume the CLI args have the same names as from_pretrained method args.
    """
    return model_cls.from_pretrained(**get_model_kwargs(model_cls, vars(cli_args)))


WorkbenchModelCollectionT = TypeVar(
    "WorkbenchModelCollectionT", bound=WorkbenchModelCollection
)


def demo_model_components_from_cli_args(
    model_cls: type[WorkbenchModelCollectionT],
    model_id: str,
    cli_args: argparse.Namespace,
) -> tuple[WorkbenchModelCollectionT, tuple[WorkbenchModel | OnDeviceModel, ...]]:
    """
    Load a collection model and its components for an on-device or local demo.

    Instantiates the collection via from_pretrained, then wraps each component
    in OnDeviceModel if on-device mode is active.

    Parameters
    ----------
    model_cls
        WorkbenchModelCollection subclass (must implement FromPretrainedProtocol).
    model_id
        Model ID string used for compilation/export naming.
    cli_args
        Command line arguments namespace (from get_on_device_demo_parser).

    Returns
    -------
    collection : WorkbenchModelCollectionT
        The instantiated collection model (from from_pretrained).
    components : tuple[WorkbenchModel | OnDeviceModel, ...]
        Model instances for each component (ordered by collection.component_names).
        Wrapped in OnDeviceModel when on-device mode is active.
    """
    assert issubclass(model_cls, FromPretrainedProtocol)
    collection = model_from_cli_args(model_cls, cli_args)
    component_names = collection.component_names
    is_on_device = "eval_mode" in cli_args and cli_args.eval_mode == EvalMode.ON_DEVICE

    res: list[WorkbenchModel | OnDeviceModel] = []
    if not is_on_device:
        res.extend([collection.components[comp_name] for comp_name in component_names])
    else:
        hub_model_ids = (
            cli_args.hub_model_id.split(",") if cli_args.hub_model_id else None
        )
        if hub_model_ids:
            assert len(hub_model_ids) == len(component_names)

        for i, comp_name in enumerate(component_names):
            component = collection.components[comp_name]
            if hub_model_ids:
                target_model = hub.get_model(hub_model_ids[i])
            else:
                cli_dict = vars(cli_args)
                additional_kwargs = dict(
                    get_model_kwargs(type(component), args_dict=cli_dict),
                    **filter_kwargs(component.get_input_spec, cli_dict),
                )
                target_model = compile_model_from_args(  # type: ignore[assignment]
                    model_id, cli_args, additional_kwargs, comp_name
                )
                assert not isinstance(target_model, list)
                print(f"Exported asset: {model_id}::{comp_name}\n")
            res.append(
                OnDeviceModel(
                    target_model,
                    list(component.get_input_spec().keys()),
                    cli_args.device,
                    cli_args.inference_options,
                )
            )

    return collection, tuple(res)


def demo_model_from_cli_args(
    model_cls: type[FromPretrainedTypeVar],
    model_id: str,
    cli_args: argparse.Namespace,
    component: str | None = None,
) -> FromPretrainedTypeVar | OnDeviceModel:
    """
    Create this model from an argparse namespace.
    Default behavior is to assume the CLI args have the same names as from_pretrained method args.

    If the model is a WorkbenchModel and an on-device demo is requested,
        the WorkbenchModel will be wrapped in an OnDeviceModel.
    """
    is_on_device = "eval_mode" in cli_args and cli_args.eval_mode == EvalMode.ON_DEVICE
    inference_model: FromPretrainedTypeVar | OnDeviceModel
    inference_model = model_from_cli_args(model_cls, cli_args)
    if is_on_device and issubclass(model_cls, WorkbenchModel):
        assert isinstance(inference_model, WorkbenchModel)
        device: hub.Device = cli_args.device
        if cli_args.hub_model_id:
            model_from_hub = hub.get_model(cli_args.hub_model_id)
            inference_model = OnDeviceModel(
                model_from_hub,
                list(inference_model.get_input_spec().keys()),
                device,
                cli_args.inference_options,
            )
        else:
            cli_dict = vars(cli_args)
            additional_kwargs = dict(
                get_model_kwargs(model_cls, args_dict=cli_dict),
                **filter_kwargs(model_cls.get_input_spec, cli_dict),
            )
            target_model = compile_model_from_args(
                model_id,
                cli_args,
                additional_kwargs,
                component,
            )
            assert isinstance(target_model, hub.Model)
            input_names = list(inference_model.get_input_spec().keys())
            inference_model = OnDeviceModel(
                target_model,
                input_names,
                device,
                inference_options=cli_args.inference_options,
            )
            print(
                f"Exported asset: {model_id}"
                + (f"::{component}" if component else "")
                + "\n"
            )

    return inference_model


def _parse_int_list(value: str) -> list[int]:
    """Parse a CLI value as a comma-separated list of ints."""
    return [int(v.strip()) for v in value.split(",")]


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    """Parse a CLI value as a comma-separated tuple of ints."""
    return tuple(int(v.strip()) for v in value.split(","))


def _resolve_param_type(
    param: inspect.Parameter,
    model_cls: type[WorkbenchModel | PrecompiledWorkbenchModel | CollectionModel],
) -> type | Callable:
    """
    Resolve a parameter annotation to a concrete type or callable for argparse.

    locate() converts a string type annotation to a class type.
    Any type can be resolved as long as it's accessible in this scope.

    TODO(#16652): This is brittle since it requires the parameter
    to be imported into that scope exactly, which may not be its
    native location.
    """
    if param.annotation is inspect.Parameter.empty:
        raise TypeError(
            f"Parameter '{param.name}' of {model_cls.__name__}.get_input_spec "
            "has no type annotation."
        )
    if isinstance(param.annotation, type):
        return param.annotation
    anno = param.annotation
    type_ = locate(anno.split(" | ", 1)[0])
    if anno == "list[int]":
        return _parse_int_list
    if anno.split(" | ", 1)[0] in ("tuple[int, int]", "tuple[int, ...]"):
        return _parse_int_tuple
    if type_ is None:
        type_ = locate(f"{model_cls.__module__}.{anno}")
    if not isinstance(type_, type):
        raise TypeError(
            f"Annotation '{anno}' for '{param.name}' did not resolve "
            f"to a type (got {type_!r})."
        )
    return type_


def get_model_input_spec_parser(
    model_cls: type[WorkbenchModel | CollectionModel],
    parser: QAIHMArgumentParser | None = None,
) -> QAIHMArgumentParser:
    """
    Generate the argument parser to get this model's input spec from an argparse namespace.
    Default behavior is to assume the CLI args have the same names as get_input_spec method args.
    """
    if not parser:
        parser = get_parser()

    input_spec_docs = {
        param.name: "\n".join(param.desc)
        for param in FunctionDoc(func=model_cls.get_input_spec)["Parameters"]
    }

    for name, param in get_params(model_cls.get_input_spec).items():
        resolved_type = _resolve_param_type(param, model_cls)
        help_str = input_spec_docs.get(
            name,
            f"For documentation, see {model_cls.__name__}::get_input_spec.",
        )
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            type=resolved_type,
            default=param.default,
            help=help_str,
        )
    return parser


def input_spec_from_cli_args(
    model: WorkbenchModel | OnDeviceModel, cli_args: argparse.Namespace
) -> InputSpec:
    """
    Create this model's input spec from an argparse namespace.
    Default behavior is to assume the CLI args have the same names as get_input_spec method args.
    Also, fetches shapes if demo is run on-device.
    """
    if isinstance(model, OnDeviceModel):
        assert "eval_mode" in cli_args and cli_args.eval_mode == EvalMode.ON_DEVICE
        producer = model.model.get_producer()
        assert isinstance(producer, hub.CompileJob)
        return cast(InputSpec, producer.shapes)
    return model.get_input_spec(**filter_kwargs(model.get_input_spec, vars(cli_args)))


def _evaluate_export_common_parser(
    model_cls: type[FromPretrainedTypeVar],
    supported_precision_runtimes: dict[Precision, list[TargetRuntime]],
    omit_precision: bool = False,
) -> QAIHMArgumentParser:
    """Common arguments between export and evaluate scripts."""
    # Set handler to resolve, to allow from_pretrained and get_input_spec
    # to have the same argument names.
    parser = get_parser(
        model_cls,
        supported_precision_runtimes,
        allow_dupe_args=True,
    )
    # Default runtime for compiled model is fixed for given model
    # Python doesn't have ordered sets, so use a dictionary to preserver order
    available_runtimes: dict[TargetRuntime, None] = {}
    for rts in supported_precision_runtimes.values():
        for rt in rts:
            available_runtimes[rt] = None

    available_runtimes_list = list(available_runtimes.keys())
    # Default is resolved dynamically in parse_args based on the chosen precision.
    add_target_runtime_arg(
        parser,
        available_target_runtimes=available_runtimes_list,
        default=None,
        helpmsg="The runtime for which to export. Default is chosen based on the precision.",
    )
    if issubclass(model_cls, FromPretrainedProtocol):
        # Skip adding CLI from model for compiled model
        # TODO: #9408 Refactor WorkbenchModel, PrecompiledWorkbenchModel to fetch
        # parameters from compiled model
        parser = get_model_cli_parser(model_cls, parser)
        parser = add_input_spec_args(model_cls, parser)  # type: ignore[arg-type]

        supported_precisions = {
            precision
            for precision, rts in supported_precision_runtimes.items()
            if len(rts) > 0
        }
        non_float_precision = _get_non_float_precision(supported_precisions)
        if not omit_precision:
            add_precision_arg(
                parser,
                supported_precisions,
                default_if_arg_explicitly_passed=non_float_precision or Precision.float,
                default=(
                    Precision.float
                    if (
                        len(supported_precisions) == 0
                        or Precision.float in supported_precisions
                    )
                    else next(iter(supported_precisions))
                ),
            )

    return parser


def add_export_function_args(
    export_fn: Callable,
    parser: QAIHMArgumentParser,
    zip_assets: bool = False,
) -> None:
    """
    Extracts the relevant inputs to the export function and
    adds them to the parser.
    """
    signature = dict(inspect.signature(export_fn).parameters)
    for key in [
        "components",
        "precision",
        "target_runtime",
        "additional_model_kwargs",
        # LLM specific args
        "model_cls",
        "position_processor_cls",
        "model_id",
        "model_name",
        "model_asset_version",
        "sub_components",
        "num_layers_per_split",
        "num_splits",
    ]:
        if key in signature:
            signature.pop(key)

    if "zip_assets" in signature:
        signature.pop("zip_assets")
        parser.add_argument(
            "--zip-assets",
            action="store_true",
            help="If set, downloaded assets are zipped.",
        )

    raw_doc = inspect.getdoc(export_fn)
    assert raw_doc is not None, "Export function must have a docstring."

    export_docs = {
        param.name: "\n".join(param.desc)
        for param in FunctionDoc(export_fn)["Parameters"]
    }

    def _get_export_help(param_name: str, default_value: Any) -> str:
        description = export_docs[param_name]
        assert description is not None, f"Input `{param_name}` must have a description."
        if default_value is True:
            description = description.replace("skips", "does")
        return description

    add_function_parser_args(signature, parser, _get_export_help)

    if "quantized_model_id" in signature:
        signature.pop("quantized_model_id")
        parser.add_argument(
            "--quantized-model-id",
            type=str,
            default=None,
            help="If set, uses this quantized model ID instead of quantizing during export. "
            "For WorkbenchModel: --quantized-model-id <id>. "
            "For CollectionModel: --quantized-model-id <id1,id2,...>. "
            "For CollectionModel, IDs must be provided in the same order as the selected components in --components.",
        )


def export_parser(
    model_cls: type[FromPretrainedProtocol],
    export_fn: Callable,
    components: list[str] | None = None,
    supported_precision_runtimes: dict[Precision, list[TargetRuntime]] | None = None,
    default_export_device: str | None = None,
    zip_assets: bool = False,
    omit_precision: bool = False,
) -> QAIHMArgumentParser:
    """
    Arg parser to be used in export scripts.

    Parameters
    ----------
    model_cls
        Class of the model to be exported. Used to add additional
        args for model instantiation.
    export_fn
        Export function to extract parameters from.
    components
        Only used for model with component and sub-component, such
        as Llama 2, 3, where two subcomponents (e.g.,
        PromptProcessor_1, TokenGenerator_1)
        are classified under one component (e.g. Llama2_Part1_Quantized).
    supported_precision_runtimes
        The list of supported (precision, runtime) pairs for this model.
    default_export_device
        Default device to set for export.
    zip_assets
        Zips downloaded assets. If set, adds --zip-assets argument to the parser.
    omit_precision
        Do not register --precision.

    Returns
    -------
    parser : QAIHMArgumentParser
        ArgumentParser object.
    """
    if supported_precision_runtimes is None:
        supported_precision_runtimes = {
            Precision.float: [TargetRuntime.TFLITE],
        }
    parser = _evaluate_export_common_parser(
        model_cls=model_cls,
        supported_precision_runtimes=supported_precision_runtimes,
        omit_precision=omit_precision,
    )
    add_export_function_args(export_fn, parser, zip_assets)
    _add_device_args(parser, default_device=default_export_device)
    if components is not None or issubclass(model_cls, CollectionModel):
        parser.add_argument(
            "--components",
            nargs="+",
            type=str,
            default=None,
            help="Which components of the model to be exported.",
        )

    return parser


def evaluate_parser(
    model_cls: type[FromPretrainedTypeVar],
    supported_dataset_classes: Sequence[type[BaseDataset]],
    supported_precision_runtimes: dict[Precision, list[TargetRuntime]] | None = None,
    uses_quantize_job: bool = True,
    num_calibration_samples: int | None = None,
    default_device: str | None = None,
) -> QAIHMArgumentParser:
    """
    Arg parser to be used in evaluate scripts.

    Parameters
    ----------
    model_cls
        Class of the model to be exported. Used to add additional args for model instantiation.
    supported_dataset_classes
        List of supported dataset classes (subclasses of BaseDataset).
    supported_precision_runtimes
        The list of supported (precision, runtime) pairs for this model.
    uses_quantize_job
        Whether this model uses quantize job to quantize the model.
    num_calibration_samples
        How many samples to calibrate on when quantizing by default.
        If not set, defers to the dataset to decide the number.
    default_device:
        The default device to use for export + eval.

    Returns
    -------
    parser : QAIHMArgumentParser
        ArgumentParser object.
    """
    if supported_precision_runtimes is None:
        supported_precision_runtimes = {Precision.float: [TargetRuntime.TFLITE]}
    parser = _evaluate_export_common_parser(
        model_cls=model_cls,
        supported_precision_runtimes=supported_precision_runtimes,
    )
    parser.set_supported_dataset_classes(supported_dataset_classes)
    parser.add_argument(
        "--compile-options",
        type=str,
        default="",
        help="Additional options to pass when submitting the compile job.",
    )
    parser.add_argument(
        "--profile-options",
        type=str,
        default="",
        help="Additional options to pass when submitting the profile job.",
    )
    if uses_quantize_job:
        parser.add_argument(
            "--quantize-options",
            type=str,
            default="",
            help="Additional options to pass when submitting the quantize job.",
        )

    _add_device_args(parser, default_device=default_device)
    if not parser._dataset_name_to_cls:
        return parser
    supported_dataset_names = list(parser._dataset_name_to_cls.keys())
    if uses_quantize_job:
        parser.add_argument(
            "--num-calibration-samples",
            type=int,
            default=num_calibration_samples,
            help="The number of calibration data samples to use for quantization.",
        )
    parser.add_argument(
        "--samples-per-job",
        type=int,
        default=None,
        help="Max size to be submitted in a single inference job.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=supported_dataset_names[0],
        choices=supported_dataset_names,
        help="Name of the dataset to use for evaluation.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to run. If set to -1, will run on full dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed to use when shuffling the data. If not set, samples data deterministically.",
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="A compiled hub model id.",
    )
    parser.add_argument(
        "--use-dataset-cache",
        action="store_true",
        help="If set, will store hub dataset ids in a local file and re-use "
        "for subsequent evaluations on the same dataset.",
    )
    if uses_quantize_job:
        parser.add_argument(
            "--compute-quant-cpu-accuracy",
            action="store_true",
            help="If flag is set, computes the accuracy of the quantized onnx model on the CPU.",
        )
    parser.add_argument(
        "--skip-device-accuracy",
        action="store_true",
        help="If flag is set, skips computing accuracy on device.",
    )
    parser.add_argument(
        "--skip-torch-accuracy",
        action="store_true",
        help="If flag is set, skips computing accuracy with the torch model.",
    )
    return parser
