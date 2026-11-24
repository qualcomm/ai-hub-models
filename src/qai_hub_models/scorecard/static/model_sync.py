# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import tempfile

import h5py
from qai_hub.util.dataset_entries_converters import (
    dataset_entries_to_h5,
    h5_to_dataset_entries,
)

from qai_hub_models.scorecard.static.model_config import ScorecardModelConfig
from qai_hub_models.utils.hub_clients import get_scorecard_client_or_raise
from qai_hub_models.utils.transpose_channel import transpose_channel_first_to_last

DEFAULT_DEPLOYMENTS = {"prod", "staging", "dev"}


def sync_model_assets(
    model_config: ScorecardModelConfig,
    deployments: str | list[str] | set[str] | None = None,
    clear_existing: bool = False,
) -> None:
    """
    Sync the reference assets (model file, dataset file) for the given model to the given list of AI Hub Workbench deployments.

    Updates automated fields in model_config by uploading the config's model & dataset to each target deployment.

    Parameters
    ----------
    model_config
        Model config to update.
    deployments
        Deployments to synchronize. If a deployment already has an asset uploaded, it is skipped.
    clear_existing
        Clears all existing automated fields before uploading (no deployments are skipped).

    Returns
    -------
    None
        Parameter model_config has updated fields.
    """
    if deployments is None:
        deployments = DEFAULT_DEPLOYMENTS.copy()
    elif isinstance(deployments, str):
        deployments = {deployments}
    elif isinstance(deployments, list):
        deployments = set(deployments)

    # Clear previous config
    if clear_existing:
        model_config.hub_model_ids_automated.clear()
        model_config.hub_input_dataset_ids_automated.clear()
        model_config.hub_input_channel_last_dataset_ids_automated.clear()

    # Only modify deployments that don't exist already in the config
    target_dataset_deployments = deployments.difference(
        model_config.hub_input_dataset_ids_automated.keys()
    )
    target_cl_dataset_deployments = deployments.difference(
        model_config.hub_input_channel_last_dataset_ids_automated.keys()
    )
    target_model_deployments = deployments.difference(
        model_config.hub_input_dataset_ids_automated.keys()
    )

    # Get prod client. The reference assets are always uploaded to prod.
    prod_client = get_scorecard_client_or_raise("prod", model_config.restrict_access)

    # Model
    if target_model_deployments:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Download model
            model = prod_client.get_model(model_config.hub_model_id)
            model_path = None

            # Upload model to each target deployment
            for deployment in target_model_deployments:
                print(f"Uploading Model to {deployment}")
                if not model_path:
                    model_path = model.download(tmpdir)
                client = get_scorecard_client_or_raise(
                    deployment, model_config.restrict_access
                )
                new_hub_model = client.upload_model(
                    model_path, f"QAIHMS::{model_config.id}"
                )
                model_config.hub_model_ids_automated[deployment] = (
                    new_hub_model.model_id
                )

            # Replace reference model with the model uploaded by the bot
            if "prod" in target_model_deployments:
                model_config.hub_model_id = model_config.hub_model_ids_automated["prod"]

    # Dataset
    if target_dataset_deployments or target_cl_dataset_deployments:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Download data to .h5 file
            dataset = prod_client.get_dataset(model_config.hub_input_dataset_id)
            dataset_h5_path = None
            dataset_name = f"QAIHMS::{model_config.id}::dataset"

            # Upload dataset to each target deployment
            for deployment in target_dataset_deployments:
                if not dataset_h5_path:
                    print(f"({model_config.id}) Downloading Reference Dataset")
                    dataset_h5_path = dataset.download(tmpdir)

                print(f"({model_config.id}) Uploading Dataset to {deployment}")
                client = get_scorecard_client_or_raise(
                    deployment, model_config.restrict_access
                )
                new_hub_dataset = client.upload_dataset(dataset_h5_path, dataset_name)
                model_config.hub_input_dataset_ids_automated[deployment] = (
                    new_hub_dataset.dataset_id
                )

            # Replace reference dataset with the dataset uploaded by the bot
            if "prod" in target_model_deployments:
                model_config.hub_input_dataset_id = (
                    model_config.hub_input_dataset_ids_automated["prod"]
                )

            # Channel last dataset
            if model_config.channel_first_inputs and target_cl_dataset_deployments:
                cl_dataset_name = f"{dataset_name}::channel_last"
                if not dataset_h5_path:
                    print(f"({model_config.id}) Downloading Reference Dataset")
                    dataset_h5_path = dataset.download(tmpdir)

                with h5py.File(dataset_h5_path, "r") as h5f:
                    print(f"({model_config.id}) Loading Reference Dataset")

                    # Load channel first dataset
                    try:
                        dataset_numpy = h5_to_dataset_entries(h5f)
                    except KeyError:
                        # TODO: Fix incorrectly added datasets for bench models.
                        # https://github.com/qcom-ai-hub/tetracode/issues/15100
                        dataset_numpy = {}

                    # Channel First -> Channel Last
                    print(
                        f"({model_config.id}) Converting Reference Dataset to Channel Last"
                    )
                    dataset_cl_numpy = transpose_channel_first_to_last(
                        model_config.channel_first_inputs, dataset_numpy
                    )
                    dataset_cl_h5_path = os.path.join(tmpdir, "___dataset_cl.h5")

                    print(
                        f"({model_config.id}) Converting Loaded Channel Last Dataset to H5"
                    )
                    with h5py.File(dataset_cl_h5_path, "w") as ds_cl_h5:
                        dataset_entries_to_h5(dataset_cl_numpy, ds_cl_h5)

                    # Upload channel last dataset to each target deployment
                    for deployment in target_cl_dataset_deployments:
                        client = get_scorecard_client_or_raise(
                            deployment, model_config.restrict_access
                        )
                        print(
                            f"({model_config.id}) Uploading Channel Last Dataset to {deployment}"
                        )
                        new_hub_cl_dataset = client.upload_dataset(
                            dataset_cl_h5_path, cl_dataset_name
                        )
                        model_config.hub_input_channel_last_dataset_ids_automated[
                            deployment
                        ] = new_hub_cl_dataset.dataset_id
