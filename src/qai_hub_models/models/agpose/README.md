# [AG-Pose: Category-level 6D object pose estimation](https://aihub.qualcomm.com/models/agpose)

AG-Pose is a category-level 6D object pose estimation model that predicts rotation, translation, and size of objects from RGB-D input. It uses DINOv2-S (ViT-S/14) as the RGB backbone and PointNet++ for 3D point cloud processing, with an Instance-Adaptive Keypoint Detector (IAKD) and Geometric-Aware Feature Aggregator (GAFA) for robust pose estimation across 6 object categories.

This is based on the implementation of AG-Pose found [here](https://github.com/Leeiieeo/AG-Pose).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/agpose).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Quick Start

Use our lightweight command-line interface to inspect and download AG-Pose:

```bash
pip install qai_hub_models_cli # (the CLI is also available with the qai-hub-models package)

# Inspect the model and list the available download options
qai-hub-models info AG-Pose

# Print performance and accuracy metrics
qai-hub-models perf AG-Pose
qai-hub-models numerics AG-Pose

# Download a ready-to-deploy asset
qai-hub-models fetch AG-Pose --runtime qnn_dlc --precision float
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
qai-hub-models install agpose
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
qai-hub-models demo agpose
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
qai-hub-models export agpose
```
Additional options are documented with the `--help` option.

## License
* The license for the original implementation of AG-Pose can be found
  [here](https://github.com/Leeiieeo/AG-Pose/blob/main/LICENSE).

## References
* [AG-Pose: Instance-Adaptive and Geometric-Aware Keypoint Learning for Category-Level 6D Object Pose Estimation](https://arxiv.org/abs/2403.19527)
* [Source Model Implementation](https://github.com/Leeiieeo/AG-Pose)

## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
