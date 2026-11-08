# [PiperTTS-EN-Amy: High-quality text-to-speech for English (Amy voice)](https://aihub.qualcomm.com/models/pipertts_en_amy)

PiperTTS is a high-quality multi-lingual text-to-speech library. This model uses the Amy English voice (medium quality). Its training data is licensed CC-BY-SA-4.0 per the mimic3-voices project (https://github.com/MycroftAI/mimic3-voices).

This is based on the implementation of PiperTTS-EN-Amy found [here](https://github.com/rhasspy/piper).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/pipertts_en_amy).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Quick Start

Use our lightweight command-line interface to inspect and download PiperTTS-EN-Amy:

```bash
pip install qai_hub_models_cli # (the CLI is also available with the qai-hub-models package)

# Inspect the model and list the available download options
qai-hub-models info PiperTTS-EN-Amy

# Print performance and accuracy metrics
qai-hub-models perf PiperTTS-EN-Amy
qai-hub-models numerics PiperTTS-EN-Amy

# Download a ready-to-deploy asset
qai-hub-models fetch PiperTTS-EN-Amy --runtime voice_ai --precision float
```
See the [CLI README](../../../../cli/README.md)
for the full list of commands and filters.

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
# NOTE: 3.10 <= PYTHON_VERSION < 3.12 is supported.
pip install qai-hub-models
qai-hub-models install pipertts_en_amy
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
qai-hub-models demo pipertts_en_amy
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
qai-hub-models export pipertts_en_amy
```
Additional options are documented with the `--help` option.

## License
* The license for the original implementation of PiperTTS-EN-Amy can be found
  [here](https://github.com/rhasspy/piper/blob/master/LICENSE.md).

## References
* [PiperTTS High-quality Multi-lingual Multi-accent Text-to-Speech](https://github.com/rhasspy/piper)
* [Source Model Implementation](https://github.com/rhasspy/piper)

## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
