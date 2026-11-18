# [First-Order-Motion-Model: Animation of Still Image from Source Video](https://aihub.qualcomm.com/models/fomm)

FOMM is a machine learning model that animates a still image to mirror the movements from a target video.

This is based on the implementation of First-Order-Motion-Model found [here](https://github.com/AliaksandrSiarohin/first-order-model/tree/master).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/fomm).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Quick Start

Use our lightweight command-line interface to inspect and download First-Order-Motion-Model:

```bash
pip install qai_hub_models_cli # (the CLI is also available with the qai-hub-models package)

# Inspect the model and list the available download options
qai-hub-models info First-Order-Motion-Model

# Print performance and accuracy metrics
qai-hub-models perf First-Order-Motion-Model
qai-hub-models numerics First-Order-Motion-Model

# Download a ready-to-deploy asset
qai-hub-models fetch First-Order-Motion-Model --runtime tflite --precision float
```
See the [CLI README](../../../../cli/README.md)
for the full list of commands and filters.

## Setup
### 1. Install System-Level Dependencies
#### Linux
```bash
sudo apt install ffmpeg
```

 #### Windows
```
winget install ffmpeg
```

### 2. Install the package
Install the base package, then use the `qai-hub-models` CLI to install this
recipe's dependencies:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.14 is supported.
pip install qai-hub-models
qai-hub-models install fomm
```

### 3. Configure Qualcomm® AI Hub Workbench
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
qai-hub-models demo fomm
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment
To run the model on Qualcomm® devices, you must export the model for use with an edge runtime such as
TensorFlow Lite, ONNX Runtime, or Qualcomm AI Engine Direct.
Use the following command to export the model:
```bash
qai-hub-models export fomm
```
Additional options are documented with the `--help` option.

## License
* The license for the original implementation of First-Order-Motion-Model can be found
  [here](https://github.com/AliaksandrSiarohin/first-order-model/blob/master/LICENSE.md).

## References
* [First Order Motion Model for Image Animation](https://arxiv.org/abs/2003.00196)
* [Source Model Implementation](https://github.com/AliaksandrSiarohin/first-order-model/tree/master)

## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
