# [PP-OCRv5: PaddleOCR v5 text detection with recognition for 7 languages for on-device OCR](https://aihub.qualcomm.com/models/pp_ocrv5)

PP-OCRv5 is a multilingual OCR pipeline with a shared text detector and language-specific recognizers. This collection bundles the detector with all seven PP-OCRv5 recognition components — Simplified Chinese (with Japanese coverage), English, Latin-script, Cyrillic / East-Slavic, Korean, Thai, and Greek — for fully on-device optical character recognition.

This is based on the implementation of PP-OCRv5 found [here](https://github.com/PaddlePaddle/PaddleOCR).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/pp_ocrv5).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Components

| Component | Description | Input resolution |
|-----------|-------------|------------------|
| `det` | Text detection (PP-HGNetV2 + DB++) | 640x640 |
| `zh_rec` | Simplified Chinese + English recognition (SVTR/CTC) | 48x320 |
| `ko_rec` | Korean + English recognition (SVTR/CTC) | 48x320 |
| `en_rec` | English recognition (SVTR/CTC) | 48x320 |
| `latin_rec` | Latin-script recognition (SVTR/CTC) | 48x320 |
| `eslav_rec` | Cyrillic / East-Slavic recognition (SVTR/CTC) | 48x320 |
| `thai_rec` | Thai recognition (SVTR/CTC) | 48x320 |
| `greek_rec` | Greek recognition (SVTR/CTC) | 48x320 |

All seven recognizers share the PP-OCRv5 SVTR/CTC architecture; each ships with
its own character dictionary baked into the exported weights.

## Setup
### 1. Install the package
Install the package via pip:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.14 is supported.
pip install "qai-hub-models[pp-ocrv5]"
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
python -m qai_hub_models.models.pp_ocrv5.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment
To run the model on Qualcomm® devices, you must export the model for use with an edge runtime such as
TensorFlow Lite, ONNX Runtime, or Qualcomm AI Engine Direct. Use the following command to export the model:
```bash
python -m qai_hub_models.models.pp_ocrv5.export
```
Additional options are documented with the `--help` option.

## Performance

Measured on a Samsung Galaxy S25 Ultra (Snapdragon 8 Elite) via Qualcomm AI Hub,
float TensorFlow Lite, all layers on the Hexagon NPU (no CPU fallback):

| Component | On-device latency |
|-----------|-------------------|
| `det` | 1.54 ms |
| `ko_rec` | 4.39 ms |
| `zh_rec` | 10.84 ms |

int8 (w8a8) quantization with a calibration set further reduces detection/recognition
latency to roughly 1.2 / 1.0 / 0.5 ms (measured via post-training quantization).

## Accuracy

Measured on **ReCTS** — real Chinese store-front / signage photographs (curved text,
occlusion, background clutter), the ReCTS subset of
[`SWHL/ChineseOCRBench`](https://huggingface.co/datasets/SWHL/ChineseOCRBench) (Apache-2.0),
n = 200 images, detector + Simplified Chinese recognizer:

| Metric | PP-OCRv5 (det + zh_rec) |
|--------|-------------------------|
| Full-string recall | 48.5% |
| Mean per-character recall | 66.04% |

## License
* The license for the original implementation of PP-OCRv5 can be found
  [here](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE).

## References
* [PaddleOCR 3.0 Technical Report](https://arxiv.org/abs/2507.05595)
* [Source Model Implementation](https://github.com/PaddlePaddle/PaddleOCR)

## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
