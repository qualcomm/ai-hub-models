# [Detectron2-Panoptic-Segmentation: Panoptic segmentation combining instance and semantic predictions on COCO](https://aihub.qualcomm.com/models/detectron2_panoptic_seg)

Detectron2-Panoptic-Segmentation is a machine learning model that performs panoptic segmentation (trained on COCO dataset), combining instance and semantic segmentation.

This is based on the implementation of Detectron2-Panoptic-Segmentation found [here](https://github.com/facebookresearch/detectron2/).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/detectron2_panoptic_seg).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Quick Start

Use our lightweight command-line interface to inspect and download Detectron2-Panoptic-Segmentation:

```bash
pip install qai_hub_models_cli # (the CLI is also available with the qai-hub-models package)

# Inspect the model and list the available download options
qai-hub-models info Detectron2-Panoptic-Segmentation

# Print performance and accuracy metrics
qai-hub-models perf Detectron2-Panoptic-Segmentation
qai-hub-models numerics Detectron2-Panoptic-Segmentation

# Download a ready-to-deploy asset
qai-hub-models fetch Detectron2-Panoptic-Segmentation --runtime tflite --precision float
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
qai-hub-models install detectron2_panoptic_seg
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
qai-hub-models demo detectron2_panoptic_seg
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
qai-hub-models export detectron2_panoptic_seg
```
Additional options are documented with the `--help` option.

## License
* The license for the original implementation of Detectron2-Panoptic-Segmentation can be found
  [here](https://github.com/facebookresearch/detectron2/blob/main/LICENSE).

## References
* [Source Model Implementation](https://github.com/facebookresearch/detectron2/)

## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
