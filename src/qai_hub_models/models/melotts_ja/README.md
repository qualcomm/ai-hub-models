# [MeloTTS-JA: High-quality text-to-speech for Japanese](https://aihub.qualcomm.com/models/melotts_ja)

MeloTTS is a high-quality multi-lingual text-to-speech library for English, Chinese, Korean, Japanese and Spanish language.

This is based on the implementation of MeloTTS-JA found [here](https://github.com/myshell-ai/MeloTTS).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/melotts_ja).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Quick Start

Use our lightweight command-line interface to inspect and download MeloTTS-JA:

```bash
pip install qai_hub_models_cli # (the CLI is also available with the qai-hub-models package)

# Inspect the model and list the available download options
qai-hub-models info MeloTTS-JA

# Print performance and accuracy metrics
qai-hub-models perf MeloTTS-JA
qai-hub-models numerics MeloTTS-JA

# Download a ready-to-deploy asset
qai-hub-models fetch MeloTTS-JA --runtime voice_ai --precision w8a16
```
See the [CLI README](../../../../cli/README.md)
for the full list of commands and filters.

## Deploying MeloTTS-JA on-device

This model is compatible with the Qualcomm Voice AI SDK. Download the SDK from the [Qualcomm Package Manager](https://qpm.qualcomm.com/#/main/tools/details/VoiceAI_TTS) to deploy this model on-device.

## Setup
### 1. Install System-Level Dependencies
#### Linux
```bash
sudo apt install libportaudio2
```

### 2. Install the package
Install the base package, then use the `qai-hub-models` CLI to install this
recipe's dependencies:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.14 is supported.
pip install qai-hub-models
qai-hub-models install melotts_ja
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
qai-hub-models demo melotts_ja
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
qai-hub-models export melotts_ja
```
Additional options are documented with the `--help` option.

## License
* The license for the original implementation of MeloTTS-JA can be found
  [here](https://github.com/myshell-ai/MeloTTS/blob/main/LICENSE).

## References
* [MeloTTS High-quality Multi-lingual Multi-accent Text-to-Speech](https://github.com/myshell-ai/MeloTTS)
* [Source Model Implementation](https://github.com/myshell-ai/MeloTTS)

## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
