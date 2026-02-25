# [MeloTTS-EN: MeloTTS is a high-quality multi-lingual text-to-speech library for English, Chinese and Spanish language](https://aihub.qualcomm.com/models/melotts_en)

MeloTTS is a high-quality multi-lingual text-to-speech library for English, Chinese and Spanish language.

This is based on the implementation of MeloTTS-EN found [here](https://github.com/myshell-ai/MeloTTS).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/melotts_en).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Setup
### 1. Install the package
Install the package via pip:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.14 is supported.
pip install --no-deps git+https://github.com/myshell-ai/MeloTTS.git
pip install "qai-hub-models[melotts-en]"
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
python -m qai_hub_models.models.melotts_en.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment
To run the model on Qualcomm® devices, you must export the model for use with an edge runtime such as
TensorFlow Lite, ONNX Runtime, or Qualcomm AI Engine Direct. Use the following command to export the model:
```bash
python -m qai_hub_models.models.melotts_en.export
```
Additional options are documented with the `--help` option.

## License
* The license for the original implementation of MeloTTS-EN can be found
  [here](https://github.com/myshell-ai/MeloTTS/blob/main/LICENSE).

## References
* [MeloTTS High-quality Multi-lingual Multi-accent Text-to-Speech](https://github.com/myshell-ai/MeloTTS)
* [Source Model Implementation](https://github.com/myshell-ai/MeloTTS)

## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
