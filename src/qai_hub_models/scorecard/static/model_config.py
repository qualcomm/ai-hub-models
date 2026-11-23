# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import contextlib
import os
from enum import Enum
from pathlib import Path
from typing import Annotated

import numpy as np
from pydantic import (
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    ValidationInfo,
    WithJsonSchema,
    field_serializer,
    model_validator,
)
from qai_hub.client import Client, CompileJob, InputSpecs, SourceModelType, UserError
from qai_hub.public_rest_api import get_dataset, get_model

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs.manifest_yaml import MODEL_DOMAIN, MODEL_TAG, MODEL_USE_CASE
from qai_hub_models.scorecard.device import ScorecardDevice
from qai_hub_models.scorecard.path_profile import (
    ScorecardInferencePathJITParseableAllList,
)
from qai_hub_models.utils.asset_loaders import EXECUTING_IN_CI_ENVIRONMENT
from qai_hub_models.utils.base_config import BaseQAIHMConfig
from qai_hub_models.utils.device import FormFactor
from qai_hub_models.utils.hub_clients import get_scorecard_client

DEFAULT_MODELS_DIR = Path(os.path.dirname(__file__)) / "models"
SCORECARD_ACCT_EMAIL = "qaihm_bot@qti.qualcomm.com"
PRIVATE_SCORECARD_ACCT_EMAIL = "qaihm_bot_private@qti.qualcomm.com"


class ScorecardModelConfig(BaseQAIHMConfig):
    class ModelInput(BaseQAIHMConfig):
        # This does 2 things:
        # * Allows "arbitrary types" (numpy dtype) to be parsed.
        # * Forbids unknown keys in parsed YAML / JSON files.
        #   This is the default for BaseQAIHMConfig, but must be re-specified here.
        model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

        # Input name
        name: str

        # Input shape
        shape: list[int]

        # Input data type
        dtype: Annotated[
            np.dtype,
            BeforeValidator(
                lambda x: np.dtype(getattr(np, x)) if isinstance(x, str) else x
            ),
            PlainSerializer(lambda x: x.name, return_type=str),
            WithJsonSchema({"type": "string"}, mode="serialization"),
        ]

    class BU(Enum):
        AI_HUB = "ai_hub"
        AUTO = "auto"
        COMPUTE = "compute"
        IOT = "iot"
        XR = "xr"

    # Model ID (same as name of config file)
    id: str

    # Model to execute, uploaded to AI Hub Workbench.
    # Must be shared with qaihm_bot@qti.qualcomm.com (or qaihm_bot_private@qti.qualcomm.com if restrict_access is True)
    hub_model_id: str

    # Example Model Input Data, uploaded as a dataset to AI Hub Workbench.
    # Must be shared with qaihm_bot@qti.qualcomm.com (or qaihm_bot_private@qti.qualcomm.com if restrict_access is True)
    hub_input_dataset_id: str

    # Type of Source Model
    type: Annotated[
        SourceModelType,
        BeforeValidator(
            lambda x: SourceModelType[x.upper()] if isinstance(x, str) else x
        ),
        PlainSerializer(lambda x: x.name, return_type=str),
    ]

    # BU Owner of this model
    bu_owner: BU = BU.COMPUTE

    # Qualcomm Sponsors (users or NTK groups)
    sponsors: list[str]

    # Why was this model added to the AI Hub Models scorecard? What is its relevance to your BU / Qualcomm?
    context: str

    # Where did this model come from? How was it and the sample data generated / exported?
    # This should have links for model code, data, and weights unless it was developed fully internally.
    source: str

    # The domain the model is used in such as computer vision, audio, etc.
    domain: MODEL_DOMAIN

    # What task the model is used to solve, such as object detection, classification, etc.
    use_case: MODEL_USE_CASE

    # Applicable model tags
    tags: list[MODEL_TAG]

    # If set, access to the model and related AI Hub Workbench jobs will be restricted to only the AI Hub team.
    # You will have access to results, but not be able to look at the jobs or models on AI Hub Workbench.
    restrict_access: bool = False

    # Valid device form factors for this model. If set, all supported scorecard devices of these form factors are tested.
    # If unspecified, enabled_devices are used instead. Cannot be specified at the same time as enabled_devices.
    #
    # See the FormFactor enum (https://github.com/search?q=repo%3Aqualcomm%2Fai-hub-models+FormFactor&type=code) for valid values.
    # "all" may also be used here if all Qualcomm hardware should be targeted.
    # All devices should only apply sparingly, as the resources required to run said models are very expensive.
    enabled_device_form_factors: list[FormFactor] | None = None

    # Device names to test. Must be the names defined in qai_hub_models/scorecard/device.py or "default"
    # If "default" is set, it resolves to DEFAULT_SCORECARD_DEVICE
    # If unspecified, enabled_device_form_factors are used instead. Cannot be specified at the same time as enabled_device_form_factors.
    enabled_devices: list[ScorecardDevice] | None = None

    # Runtimes we want to test this model on.
    # If a profile path is not on this list, it will always be skipped for this model regardless of scorecard settings.
    # See valid values in qai_hub_Models/scorecard/path_profile.py
    #
    # The default is "all", which enables all JIT (on-device prepare) compile paths.
    enabled_profile_runtimes: ScorecardInferencePathJITParseableAllList = Field(
        default_factory=lambda: ScorecardInferencePathJITParseableAllList.default()
    )

    # If set, this model is skipped entirely. This should be set to a string reason if the model is to be disabled.
    disabled_reason: str | None = None

    # The precision that the model's graph uses.
    # If float, skips this model on chipsets that support only quantized compute.
    # See qai_hub_models.common.py::Precision for valid options
    precision: Precision = Precision.float

    # Model input spec (applicable only for TorchScript models)
    input_specs: list[ScorecardModelConfig.ModelInput] | None = None

    # Model output names (applicable only for TorchScript models)
    output_names: list[str] | None = None

    # Inputs that are in channel-first (NCHW) format.
    # In other words, these inputs have a channel dimension that immediately follows the batch dimension.
    #
    # When compiling for some runtimes, to optimize performance,
    # the shapes of these inputs are changed to channel-last (NHWC).
    # In channel-last format, the channel dimension is the rightmost (last) dimension.
    channel_first_inputs: list[str] = Field(default_factory=(list))

    # Outputs that are in channel-first (NCHW) format.
    # In other words, these outputs have a channel dimension that immediately follows the batch dimension.
    #
    # When compiling for some runtimes, to optimize performance,
    # the shapes of these outputs are changed to channel-last (NHWC).
    # In channel-last format, the channel dimension is the rightmost (last) dimension.
    channel_first_outputs: list[str] = Field(default_factory=(list))

    # Extra target-specific options to pass when compiling the model
    extra_compile_options: dict[TargetRuntime, list[str]] = Field(default_factory=dict)

    # Model IDs by AI Hub Workbench deployment-- Map<Workbench deployment name : Model ID>.
    # This field is managed by Continuous Integration and should not be modified by users.
    hub_model_ids_automated: dict[str, str] = Field(default_factory=dict)

    # Example Inputs by AI Hub Workbench deployment-- Map<Workbench deployment name : Dataset ID>.
    # This field is managed by Continuous Integration and should not be modified by users.
    hub_input_dataset_ids_automated: dict[str, str] = Field(default_factory=dict)

    # Example Inputs by AI Hub Workbench deployment-- Map<Workbench deployment name : Dataset ID>.
    # channel_first -> channel last transform is applied on inputs found in self.channel_first_inputs.
    #
    # This field is managed by Continuous Integration and should not be modified by users.
    hub_input_channel_last_dataset_ids_automated: dict[str, str] = Field(
        default_factory=dict
    )

    @property
    def devices(self) -> list[ScorecardDevice]:
        if self.enabled_device_form_factors:
            return ScorecardDevice.all_devices(
                form_factors=self.enabled_device_form_factors
            )
        assert self.enabled_devices
        return self.enabled_devices

    def get_hub_api_input_specs(self) -> InputSpecs | None:
        return (
            {x.name: (tuple(x.shape), x.dtype.name) for x in self.input_specs}
            if self.input_specs
            else None
        )

    @field_serializer("enabled_devices")
    def serialize_enabled_devices(
        self, enabled_devices: list[ScorecardDevice]
    ) -> list[str]:
        return [
            "default" if device.is_default else str(device)
            for device in enabled_devices
        ]

    @classmethod
    def from_scorecard_model_id(
        cls, model_id: str, models_dir: Path = DEFAULT_MODELS_DIR
    ) -> ScorecardModelConfig:
        path = models_dir / f"{model_id}.yaml"
        if not path.exists():
            raise ValueError(f"Could not find config for model {model_id}")
        return cls.from_yaml(path)

    def to_scorecard_yaml(self, models_dir: Path = DEFAULT_MODELS_DIR) -> bool:
        return self.to_yaml(models_dir / f"{self.id}.yaml")

    @model_validator(mode="after")
    def check_fields(self, info: ValidationInfo) -> ScorecardModelConfig:
        # Whether to validate Hub Assets. This isn't validated by default because it's expensive (several calls to Hub API).
        validate_hub_assets: bool = info.context is not None and bool(
            info.context.get("validate_hub_assets", False)
        )

        if self.type in [SourceModelType.ONNX, SourceModelType.AIMET_ONNX]:
            for attr in ["input_specs", "output_names"]:
                if getattr(self, attr) is not None:
                    raise ValueError(
                        f"{attr} should not be defined if source model type is ONNX."
                    )

        elif self.type == SourceModelType.TORCHSCRIPT:
            for attr in ["input_specs", "output_names"]:
                if getattr(self, attr) is None:
                    raise ValueError(
                        f"{attr} must be defined if source model type is Torchscript."
                    )

        else:
            raise ValueError(f"Unsupported source model type: {self.type.name}")

        if (
            self.hub_input_dataset_ids_automated.keys()
            != self.hub_model_ids_automated.keys()
        ):
            raise ValueError(
                "Automated dataset IDs and model IDs must exist for the same set of Hub deployments."
            )

        if not self.enabled_device_form_factors and not self.enabled_devices:
            raise ValueError("form_factors or enabled_devices must be set")
        if self.enabled_device_form_factors and self.enabled_devices:
            raise ValueError(
                "Either form_factors or enabled_devices must be set, but not both"
            )

        if self.enabled_devices and len(self.enabled_devices) != len(
            set(self.enabled_devices)
        ):
            raise ValueError("enabled_devices has duplicates.")

        if not validate_hub_assets:
            return self

        # Collect the list of hub deployments we need access to for testing
        hub_client_names = {"prod"}.union(self.hub_input_dataset_ids_automated.keys())

        # For each deployment, if a client configuration exists in this process' environment, get a config for it.
        # Some deployment credentials may not be available in this environment. If credentials aren't available,
        # we don't run validation on any assets related to that hub deployment.
        hub_clients = {
            x: get_scorecard_client(x, self.restrict_access) for x in hub_client_names
        }
        scorecard_account_email = (
            PRIVATE_SCORECARD_ACCT_EMAIL
            if self.restrict_access
            else SCORECARD_ACCT_EMAIL
        )

        def _verify_model(model_id: str, deployment: str, client: Client) -> str | None:
            """
            Validate the model aligns with this config.

            This is a "do the best we can given the information we have" validator. To fully validate a model against this config,
            one would need to download & process each model to extract I/O shapes. Doing so is too expensive for this function.
            """
            try:
                model_pb = get_model(client.config, model_id)
                model = client._make_model(model_pb)
            except UserError:
                if EXECUTING_IN_CI_ENVIRONMENT:
                    # CI uses the bot account
                    return f"Model {model_id} must be shared with {scorecard_account_email} in AI Hub Workbench deployment {deployment}"
                # Don't validate further, no access
                return None

            sharing_emails: list[str] | None = None
            with contextlib.suppress(
                UserError
            ):  # If UserError is thrown, no access to this API, which means the current user does not own the model
                sharing_emails = model.get_sharing()

            if (
                not EXECUTING_IN_CI_ENVIRONMENT  # CI uses the bot account and would fail when calling get_model
                and model_pb.owner.email != scorecard_account_email
                # Don't validatre sharing_emails if we can't access it.
                # Otherwise make sure the model is shared with the bot account.
                and (sharing_emails and scorecard_account_email not in sharing_emails)
            ):
                return f"Model {model_id} must be shared with {scorecard_account_email} in AI Hub Workbench deployment {deployment}"

            if model.model_type != self.type:
                return f"Model type {self.type} is not the same as type as model {model_id} ({model.model_type}) in AI Hub Workbench deployment {deployment}"

            # If the model was compiled with AI Hub Workbench, we have some additional I/O shape info accessible to validate with
            if producer_job := model.get_producer():
                assert isinstance(producer_job, CompileJob)
                target_shapes = producer_job.get_target_shapes()
                assert target_shapes is not None
                for input_name in self.channel_first_inputs:
                    if input_name not in target_shapes:
                        return f"Input {input_name} is defined in channel_first_inputs, but does not exist in the ONNX model."

                if self.precision != Precision.float:
                    for input_name, input_value in target_shapes.items():
                        assert isinstance(input_value, tuple)
                        if input_value[1] in ["int32", "float32"]:
                            return f"Model is marked quantized, but input {input_name} has type {input_value[1]}"
            return None

        def _verify_dataset(
            dataset_id: str, deployment: str, client: Client
        ) -> str | None:
            """
            Validate the dataset aligns with this config.

            This is a "do the best we can given the information we have" validator. To fully validate a dataset against this config,
            one would need to download & process each dataset to extract I/O shapes. Doing so is too expensive for this function.
            """
            try:
                dataset_pb = get_dataset(client.config, dataset_id)
                dataset = client._make_dataset(dataset_pb)
            except UserError:
                if EXECUTING_IN_CI_ENVIRONMENT:
                    # CI uses the bot account
                    return f"Dataset {dataset_id} must be shared with {scorecard_account_email} on deployment {deployment}"
                # Don't validate further, no access
                return None

            sharing_emails: list[str] | None = None
            with contextlib.suppress(
                UserError
            ):  # If UserError is thrown, no access to this API, which means the current user does not own the dataset
                sharing_emails = dataset.get_sharing()

            if (
                not EXECUTING_IN_CI_ENVIRONMENT  # CI uses the bot account and would fail when calling get_dataset
                and dataset_pb.owner.email != scorecard_account_email
                # Don't validatre sharing_emails if we can't access it.
                # Otherwise make sure the model is shared with the bot account.
                and (sharing_emails and scorecard_account_email not in sharing_emails)
            ):
                return f"Dataset {dataset_id} must be shared with {scorecard_account_email} on deployment {deployment}"

            return None

        # If we have access to prod, verify we have access to the prod model & dataset IDs in this config.
        if client := hub_clients.get("prod"):
            if bad_val := _verify_model(self.hub_model_id, "prod", client):
                raise ValueError(bad_val)
            if bad_val := _verify_dataset(self.hub_input_dataset_id, "prod", client):
                raise ValueError(bad_val)

        # If we have access to a hub deployment, verify we have access to the
        # models in the config for that deployment.
        for deployment_name, model_id in self.hub_model_ids_automated.items():
            if (client := hub_clients.get(deployment_name)) and (
                bad_val := _verify_model(model_id, deployment_name, client)
            ):
                raise ValueError(bad_val)

        # If we have access to a hub deployment, verify we have access to the
        # datasets in the config for that deployment.
        for (
            deployment_name,
            dataset_id,
        ) in self.hub_input_dataset_ids_automated.items():
            if (client := hub_clients.get(deployment_name)) and (
                bad_val := _verify_dataset(dataset_id, deployment_name, client)
            ):
                raise ValueError(bad_val)

        # If we have access to a hub deployment, verify we have access to the
        # datasets in the config for that deployment.
        for (
            deployment_name,
            dataset_id,
        ) in self.hub_input_channel_last_dataset_ids_automated.items():
            if (client := hub_clients.get(deployment_name)) and (
                bad_val := _verify_dataset(dataset_id, deployment_name, client)
            ):
                raise ValueError(bad_val)

        return self
