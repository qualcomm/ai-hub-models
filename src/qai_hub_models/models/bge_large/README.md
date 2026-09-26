> [!WARNING]
> This model is not published. Use with caution; it may not meet performance/accuracy standards and may not support some runtimes or chipsets/devices. We do not provide support for unpublished models. If this model was previously published, use earlier releases.

# [BGE-Large: High-accuracy sentence embedding model for semantic search and retrieval](https://aihub.qualcomm.com/models/bge_large)

BGE-Large-EN-v1.5 maps sentences to a 1024-dimensional dense vector space. A BERT-large based text encoder, it ranks among the strongest open embedding models on retrieval and sentence similarity benchmarks while remaining small enough to run on mobile devices.

This is based on the implementation of BGE-Large found [here](https://huggingface.co/BAAI/bge-large-en-v1.5).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/bge_large).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Quick Start

Use our lightweight command-line interface to inspect and download BGE-Large:

```bash
pip install qai_hub_models_cli # (the CLI is also available with the qai-hub-models package)

# Inspect the model and list the available download options
qai-hub-models info BGE-Large

# Print performance and accuracy metrics
qai-hub-models perf BGE-Large
qai-hub-models numerics BGE-Large

# Download a ready-to-deploy asset
qai-hub-models fetch BGE-Large --runtime tflite --precision float
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
qai-hub-models install bge_large
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
qai-hub-models demo bge_large
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
qai-hub-models export bge_large
```
Additional options are documented with the `--help` option.

## License
* The license for the original implementation of BGE-Large can be found
  [here](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md).

## References
* [C-Pack: Packed Resources For General Chinese Embeddings](https://arxiv.org/abs/2309.07597)
* [Source Model Implementation](https://huggingface.co/BAAI/bge-large-en-v1.5)

## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
