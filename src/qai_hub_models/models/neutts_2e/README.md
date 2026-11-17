# [NeuTTS-2E: On-device emotional text-to-speech language model with seven emotions](https://aihub.qualcomm.com/models/neutts_2e)

NeuTTS-2E is a compact on-device emotional text-to-speech language model from Neuphonic. It pairs a 236M-parameter Qwen3 causal LM backbone with the NeuCodec 50Hz neural audio codec, and produces natural-sounding 24kHz English speech for four fixed speakers across seven emotions. There is no voice cloning, as the four speakers ship as pre-encoded reference codes, so no reference audio is read and the codec's encoder is never called. The backbone exports as two graphs (a 128-token prefill and a single-token decode) linked into one weight-shared context binary; prompt assembly, emotion conditioning, sampling and NeuCodec decode run on CPU in the app.

This is based on the implementation of NeuTTS-2E found [here](https://github.com/neuphonic/neutts).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/neutts_2e).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Quick Start

Use our lightweight command-line interface to inspect and download NeuTTS-2E:

```bash
pip install qai_hub_models_cli # (the CLI is also available with the qai-hub-models package)

# Inspect the model and list the available download options
qai-hub-models info NeuTTS-2E

# Print performance and accuracy metrics
qai-hub-models perf NeuTTS-2E
qai-hub-models numerics NeuTTS-2E

# Download a ready-to-deploy asset
qai-hub-models fetch NeuTTS-2E --runtime qnn_context_binary --precision float
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
qai-hub-models install neutts_2e
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
qai-hub-models demo neutts_2e
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
qai-hub-models export neutts_2e
```
Additional options are documented with the `--help` option.

## License
* The license for the original implementation of NeuTTS-2E can be found
  [here](https://github.com/neuphonic/neutts/blob/ac69851f28fc63a487917e7c2e27f0d75c759cba/LICENSE).

## References
* [NeuCodec Neural Audio Codec for Speech Language Models](https://arxiv.org/abs/2509.09550)
* [Source Model Implementation](https://github.com/neuphonic/neutts)

## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
