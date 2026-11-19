# [MobileFaceNet: Lightweight real-time face verification on-device](https://aihub.qualcomm.com/models/mobile_facenet)

MobileFaceNet is an efficient CNN that maps a 112x112 face image to a compact 128-dimensional embedding. Two embeddings are compared via cosine similarity to determine whether they belong to the same person, achieving 99.48% accuracy on the LFW benchmark. The model uses depthwise-separable convolutions and inverted residual blocks (MobileNetV2-style) to stay under 1M parameters, making it well-suited for real-time face verification on mobile and edge devices. Trained with ArcFace loss on MS-Celeb-1M. This recipe runs as a two-component pipeline. A RetinaFace detector (640x640) finds the face, then MobileFaceNet (112x112) computes the embedding. Both are compiled and profiled, so the reported numbers cover the full pipeline. LFW accuracy uses the dataset's own landmarks, so it does not depend on the detector.

This is based on the implementation of MobileFaceNet found [here](https://github.com/foamliu/MobileFaceNet).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/mobile_facenet).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Quick Start

Use our lightweight command-line interface to inspect and download MobileFaceNet:

```bash
pip install qai_hub_models_cli # (the CLI is also available with the qai-hub-models package)

# Inspect the model and list the available download options
qai-hub-models info MobileFaceNet

# Print performance and accuracy metrics
qai-hub-models perf MobileFaceNet
qai-hub-models numerics MobileFaceNet

# Download a ready-to-deploy asset
qai-hub-models fetch MobileFaceNet --runtime tflite --precision float
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
qai-hub-models install mobile_facenet
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
qai-hub-models demo mobile_facenet
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
qai-hub-models export mobile_facenet
```
Additional options are documented with the `--help` option.

## License
* The license for the original implementation of MobileFaceNet can be found
  [here](https://github.com/foamliu/MobileFaceNet/blob/master/LICENSE).

## References
* [Source Model Implementation](https://github.com/foamliu/MobileFaceNet)

## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
