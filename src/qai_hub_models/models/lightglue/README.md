> [!WARNING]
> This model is not published. Use with caution; it may not meet performance/accuracy standards and may not support some runtimes or chipsets/devices. We do not provide support for unpublished models. If this model was previously published, use earlier releases.

# [LightGlue: Lightweight attention-based matcher for sparse local image features](https://aihub.qualcomm.com/models/lightglue)

LightGlue is a deep neural network that matches sparse local features across image pairs. It replaces the traditional combination of nearest-neighbour matching and robust geometric fitting with a fast, lightweight attention-based model, performing both matching and a learned pruning of non-matchable points in a single forward pass. LightGlue consumes keypoints and descriptors already produced by a separate feature extractor (SuperPoint) and outputs point correspondences between the two images. This recipe pairs LightGlue's official checkpoint (trained against Magic-Leap's SuperPoint) with this repo's MIT-licensed SuperPoint port, so reported accuracy is directionally meaningful but not a direct reproduction of the paper's figures.

This is based on the implementation of LightGlue found [here](https://github.com/cvg/LightGlue).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/lightglue).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Quick Start

Use our lightweight command-line interface to inspect and download LightGlue:

```bash
pip install qai_hub_models_cli # (the CLI is also available with the qai-hub-models package)

# Inspect the model and list the available download options
qai-hub-models info LightGlue

# Print performance and accuracy metrics
qai-hub-models perf LightGlue
qai-hub-models numerics LightGlue

# Download a ready-to-deploy asset
qai-hub-models fetch LightGlue --runtime tflite --precision float
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
qai-hub-models install lightglue
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
qai-hub-models demo lightglue
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
qai-hub-models export lightglue
```
Additional options are documented with the `--help` option.

## License
* The license for the original implementation of LightGlue can be found
  [here](https://github.com/cvg/LightGlue/blob/main/LICENSE).

## References
* [LightGlue: Local Feature Matching at Light Speed](https://arxiv.org/abs/2306.13643)
* [Source Model Implementation](https://github.com/cvg/LightGlue)

## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
