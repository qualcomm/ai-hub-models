# [EdgeTAM: Real-time on-device track-anything video segmentation and tracking](https://aihub.qualcomm.com/models/edgetam)

EdgeTAM is a lightweight and efficient version of SAM 2, optimized for on-device tracking and segmenting any object in videos and low memory usage by using a novel 2D Spatial Perceiver and RepViT backbone.

This is based on the implementation of EdgeTAM found [here](https://github.com/facebookresearch/EdgeTAM).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/edgetam).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Setup
### 1. Install the package
Install the package via pip:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.14 is supported.
pip install "qai-hub-models[edgetam]" git+https://github.com/facebookresearch/EdgeTAM.git@a1209a454c9950d531498074a95ecf3a3ba02dfd
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
python -m qai_hub_models.models.edgetam.demo { --quantize w8a8 }
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment
To run the model on Qualcomm® devices, you must export the model for use with an edge runtime such as
TensorFlow Lite, ONNX Runtime, or Qualcomm AI Engine Direct. Use the following command to export the model:
```bash
python -m qai_hub_models.models.edgetam.export { --quantize w8a8 }
```
Additional options are documented with the `--help` option.

## License
* The license for the original implementation of EdgeTAM can be found
  [here](https://github.com/facebookresearch/EdgeTAM/blob/main/LICENSE).

## References
* [EdgeTAM: On-Device Track Anything Model](https://arxiv.org/abs/2501.07256)
* [Source Model Implementation](https://github.com/facebookresearch/EdgeTAM)

## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
