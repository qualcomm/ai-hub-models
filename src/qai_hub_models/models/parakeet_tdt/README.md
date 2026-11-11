# [Parakeet-TDT-0.6B-v3: Multilingual automatic speech recognition (ASR) model supporting English and 25 European languages with Token-and-Duration Transducer (TDT) architecture](https://aihub.qualcomm.com/models/parakeet_tdt)

Parakeet-TDT-0.6B-v3 is a 600M-parameter ASR model by NVIDIA using TDT architecture with a FastConformer encoder and 2-layer LSTM predictor, split into Encoder, Decoder, and Joint components for efficient on-device inference across English and 25 European languages.

This is based on the implementation of Parakeet-TDT-0.6B-v3 found [here](https://github.com/NVIDIA/NeMo).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/parakeet_tdt).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Quick Start

Use our lightweight command-line interface to inspect and download Parakeet-TDT-0.6B-v3:

```bash
pip install qai_hub_models_cli # (the CLI is also available with the qai-hub-models package)

# Inspect the model and list the available download options
qai-hub-models info Parakeet-TDT-0.6B-v3

# Print performance and accuracy metrics
qai-hub-models perf Parakeet-TDT-0.6B-v3
qai-hub-models numerics Parakeet-TDT-0.6B-v3

# Download a ready-to-deploy asset
qai-hub-models fetch Parakeet-TDT-0.6B-v3 --runtime qnn_dlc --precision float
```
See the [CLI README](../../../../cli/README.md)
for the full list of commands and filters.

## Deploying Parakeet-TDT-0.6B-v3 on-device

This model is compatible with the Qualcomm Voice AI SDK. Download the SDK from the [Qualcomm Package Manager](https://qpm.qualcomm.com/#/main/tools/details/VoiceAI_ASR) to deploy this model on-device.

## Setup
### 1. Install the package
Install the base package, then use the `qai-hub-models` CLI to install this
recipe's dependencies:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.14 is supported.
pip install qai-hub-models
qai-hub-models install parakeet_tdt
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
qai-hub-models demo parakeet_tdt
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
qai-hub-models export parakeet_tdt
```
Additional options are documented with the `--help` option.

## License
* The license for the original implementation of Parakeet-TDT-0.6B-v3 can be found
  [here](https://creativecommons.org/licenses/by/4.0/legalcode.en).

## References
* [Canary-1B-v2 & Parakeet-TDT-0.6B-v3: Efficient and High-Performance Models for Multilingual ASR and AST](https://arxiv.org/abs/2509.14128)
* [Source Model Implementation](https://github.com/NVIDIA/NeMo)

## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
