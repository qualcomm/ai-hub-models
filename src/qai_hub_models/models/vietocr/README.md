# [VietOCR: Vietnamese text recognition with a vgg19_bn CNN backbone and transformer decoder](https://aihub.qualcomm.com/models/vietocr)

VietOCR is a text recognition model for Vietnamese, covering the full precomposed tone character set. It pairs a vgg19_bn CNN backbone with a transformer sequence decoder. This contribution exports the recognition CNN backbone for on-device deployment.

This is based on the implementation of VietOCR found [here](https://github.com/pbcquoc/vietocr).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/vietocr).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Setup
### 1. Install the package
Install the package via pip:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.14 is supported.
pip install "qai-hub-models[vietocr]"
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
python -m qai_hub_models.models.vietocr.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment
To run the model on Qualcomm® devices, you must export the model for use with an edge runtime such as
TensorFlow Lite, ONNX Runtime, or Qualcomm AI Engine Direct. Use the following command to export the model:
```bash
python -m qai_hub_models.models.vietocr.export
```
Additional options are documented with the `--help` option.

## Scope of this contribution

Vietnamese text recognition is currently absent from the AI Hub catalog. VietOCR is a
widely used, Apache-2.0, PyTorch-native Vietnamese recognizer, which makes it a good fit
for the standard trace-and-compile export path. This contribution covers the recognition
**CNN backbone** (vgg19_bn); the transformer decoder is a planned follow-up component.

## Performance

Measured on a Samsung Galaxy S25 Ultra (Snapdragon 8 Elite) via Qualcomm AI Hub,
float precision, all layers on the Hexagon NPU (no CPU fallback):

| Component | On-device latency | NPU layer coverage |
|-----------|-------------------|--------------------|
| vgg19_bn backbone | 4.48 ms | 26 / 26 (100%) |

An end-to-end recognition accuracy number is not reported here because this contribution
covers only the CNN backbone; accuracy will be reported alongside the decoder follow-up.

## Engineering note

The original backbone tail uses `permute(-1, 0, 1)` and `transpose(-1, -2).flatten(2)`.
Negative permutation axes and the implied dynamic reshape do not export to a static
on-device graph. The tail is rebuilt with equivalent static, positive-axis operations
(`transpose(2, 3)`, `permute(2, 0, 1)`), preserving semantics while producing a fully
static graph. See `model.py` and `test.py`.

## License
* The license for the original implementation of VietOCR can be found
  [here](https://github.com/pbcquoc/vietocr/blob/master/LICENSE).

## References
* [Source Model Implementation](https://github.com/pbcquoc/vietocr)

## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
