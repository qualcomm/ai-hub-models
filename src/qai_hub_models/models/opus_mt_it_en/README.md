# [OpusMT-It-En: OpusMT Italian to English neural machine translation model based on MarianMT transformer architecture](https://aihub.qualcomm.com/models/opus_mt_it_en)

OpusMT Italian to English translation model is a state-of-the-art neural machine translation system designed for translating Italian text into English. This model is based on the Marian transformer architecture and has been optimized for edge inference by splitting into encoder and decoder components with modified attention mechanisms. It exhibits robust performance for real-world translation tasks, making it highly reliable for practical applications. The model supports input sequences up to 256 tokens and can generate English translations with high accuracy.

This is based on the implementation of OpusMT-It-En found [here](https://github.com/Helsinki-NLP/Opus-MT).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/opus_mt_it_en).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Quick Start

Use our lightweight command-line interface to inspect and download OpusMT-It-En:

```bash
pip install qai_hub_models_cli # (the CLI is also available with the qai-hub-models package)

# Inspect the model and list the available download options
qai-hub-models info OpusMT-It-En

# Print performance and accuracy metrics
qai-hub-models perf OpusMT-It-En
qai-hub-models numerics OpusMT-It-En

# Download a ready-to-deploy asset
qai-hub-models fetch OpusMT-It-En --runtime voice_ai --precision w8a16
```
See the [CLI README](../../../../cli/README.md)
for the full list of commands and filters.

## Deploying OpusMT-It-En on-device

This model is compatible with the Qualcomm Voice AI SDK. Download the SDK from the [Qualcomm Package Manager](https://qpm.qualcomm.com/#/main/tools/details/VoiceAI_Translation) to deploy this model on-device.

## Setup
### 1. Install the package
Install the base package, then use the `qai-hub-models` CLI to install this
recipe's dependencies:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.14 is supported.
pip install qai-hub-models
qai-hub-models install opus_mt_it_en
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
qai-hub-models demo opus_mt_it_en
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
qai-hub-models export opus_mt_it_en
```
Additional options are documented with the `--help` option.

## License
* The license for the original implementation of OpusMT-It-En can be found
  [here](https://github.com/Helsinki-NLP/Opus-MT/blob/master/LICENSE).

## References
* [OPUS-MT – Building open translation services for the World](https://aclanthology.org/2020.eamt-1.61/)
* [Source Model Implementation](https://github.com/Helsinki-NLP/Opus-MT)

## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
