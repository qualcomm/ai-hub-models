# [License-Plates-Detector-Model: YOLOv8-based detector for UAE licence plates — characters, digits, and emirate identifiers](https://aihub.qualcomm.com/models/lpdm)

A YOLOv8-nano object detection model fine-tuned on the UAE licence-plate dataset (Roboflow car-plate-k6xij v26). The model detects and classifies 51 object classes: digits 0-9, letters A-Z, UAE emirate region identifiers and a plate bounding-box class for the full plate region.

This is based on the implementation of License-Plates-Detector-Model found [here](https://github.com/3bdul1ah/License-Plates-Detector-Model).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/lpdm).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Quick Start

Use our lightweight command-line interface to inspect License-Plates-Detector-Model:

```bash
pip install qai_hub_models_cli # (the CLI is also available with the qai-hub-models package)

# Inspect the model's metadata
qai-hub-models info License-Plates-Detector-Model

# Print performance and accuracy metrics
qai-hub-models perf License-Plates-Detector-Model
qai-hub-models numerics License-Plates-Detector-Model

# Pre-exported assets are not available to download for this model due to
# licensing restrictions. Continue to the next section to export it yourself.
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
qai-hub-models install lpdm
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
qai-hub-models demo lpdm
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
qai-hub-models export lpdm
```
Additional options are documented with the `--help` option.

## License
* The license for the original implementation of License-Plates-Detector-Model can be found
  [here](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

## References
* [Ultralytics YOLOv8 Docs: Object Detection](https://docs.ultralytics.com/tasks/detect/)
* [Source Model Implementation](https://github.com/3bdul1ah/License-Plates-Detector-Model)

## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
