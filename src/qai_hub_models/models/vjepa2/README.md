# [V-JEPA-2: Self-supervised video representation backbone](https://aihub.qualcomm.com/models/vjepa2)

V-JEPA 2 (Video Joint-Embedding Predictive Architecture 2) is a self-supervised video encoder from Meta AI that uses a ViT (Vision Transformer) backbone to produce per-token video embeddings. It is used as a frozen backbone for downstream video understanding tasks such as action recognition and video retrieval.

This is based on the implementation of V-JEPA-2 found [here](https://github.com/facebookresearch/vjepa2).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/vjepa2).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Quick Start

Use our lightweight command-line interface to inspect and download V-JEPA-2:

```bash
pip install qai_hub_models_cli # (the CLI is also available with the qai-hub-models package)

# Inspect the model and list the available download options
qai-hub-models info V-JEPA-2

# Print performance and accuracy metrics
qai-hub-models perf V-JEPA-2
qai-hub-models numerics V-JEPA-2

# Download a ready-to-deploy asset
qai-hub-models fetch V-JEPA-2 --runtime qnn_dlc --precision float
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
qai-hub-models install vjepa2
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
qai-hub-models demo vjepa2
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
qai-hub-models export vjepa2
```
Additional options are documented with the `--help` option.

## License
* The license for the original implementation of V-JEPA-2 can be found
  [here](https://github.com/facebookresearch/vjepa2/blob/main/LICENSE).

## References
* [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985)
* [Source Model Implementation](https://github.com/facebookresearch/vjepa2)

## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
