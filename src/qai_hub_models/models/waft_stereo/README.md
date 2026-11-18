# [WAFT-Stereo: Warping-based iterative stereo matching network that avoids constructing a cost volume](https://aihub.qualcomm.com/models/waft_stereo)

WAFT-Stereo estimates per-pixel disparity from a rectified stereo pair using warping-based iterative refinement instead of a 4D cost volume.

This is based on the implementation of WAFT-Stereo found [here](https://github.com/princeton-vl/WAFT-Stereo).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/waft_stereo).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Quick Start

Use our lightweight command-line interface to inspect and download WAFT-Stereo:

```bash
pip install qai_hub_models_cli # (the CLI is also available with the qai-hub-models package)

# Inspect the model and list the available download options
qai-hub-models info WAFT-Stereo

# Print performance and accuracy metrics
qai-hub-models perf WAFT-Stereo
qai-hub-models numerics WAFT-Stereo

# Download a ready-to-deploy asset
qai-hub-models fetch WAFT-Stereo --runtime onnx --precision float
```
See the [CLI README](../../../../cli/README.md)
for the full list of commands and filters.

## Setup
### 1. Install the package
Install the base package, then use the `qai-hub-models` CLI to install this
recipe's dependencies:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.14 is supported.
pip install qai-hub-models
qai-hub-models install waft_stereo
```

### 2. Configure Qualcomm® AI Hub Workbench
Sign-in to [Qualcomm® AI Hub Workbench](https://workbench.aihub.qualcomm.com/) with your
Qualcomm® ID. Once signed in navigate to `Account -> Settings -> API Token`.

With this API token, you can configure your client to run models on the cloud
hosted devices.
```bash
qai-hub configure --api_token API_TOKEN
```
Navigate to [docs](https://workbench.aihub.qualcomm.com/docs/) for more information.

## Run CLI Demo
Run the following simple CLI demo to verify the model is working end to end:

```bash
qai-hub-models demo waft_stereo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

By default, the demo will run locally in PyTorch. Pass `--eval-mode on-device` to run the model on a cloud-hosted target device.

## Export for on-device deployment
To run the model on Qualcomm® devices, you must export the model for use with an edge runtime such as
TensorFlow Lite, ONNX Runtime, or Qualcomm AI Engine Direct.
Use the following command to export the model:
```bash
qai-hub-models export waft_stereo
```
Additional options are documented with the `--help` option.

## License
* The license for the original implementation of WAFT-Stereo can be found
  [here](https://github.com/princeton-vl/WAFT-Stereo/blob/main/LICENSE).

## References
* [WAFT-Stereo: Warping-Alone Field Transforms for Stereo Matching](https://arxiv.org/abs/2603.24836)
* [Source Model Implementation](https://github.com/princeton-vl/WAFT-Stereo)

## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
