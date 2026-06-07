# [PhoWhisper-Small: Vietnamese-specialized automatic speech recognition (ASR) model, fine-tuned from Whisper-Small for accurate on-device transcription of Vietnamese speech](https://aihub.qualcomm.com/models/phowhisper_small)

PhoWhisper-Small is a Vietnamese fine-tune of OpenAI's Whisper-Small ASR (Automatic Speech Recognition) model. It shares the same transformer encoder-decoder architecture as Whisper-Small but is trained on a large, diverse Vietnamese speech corpus covering multiple dialects, substantially improving Vietnamese transcription accuracy over the generic multilingual checkpoint. It fills the gap of having no Vietnamese-specialized ASR model in the catalog. Like Whisper-Small, it has been optimized for edge inference by replacing Multi-Head Attention (MHA) with Single-Head Attention (SHA) and linear layers with convolutional (conv) layers, and it transcribes audio clips up to 30 seconds long. Time to the first token is the encoder's latency, while time to each additional token is decoder's latency, where we assume a max decoded length specified below.

On the Vietnamese FLEURS test set, PhoWhisper-Small attains a word error rate (WER) of 11.0%, compared with 21.4% for the generic Whisper-Small checkpoint.

This is based on the implementation of PhoWhisper found [here](https://github.com/VinAIResearch/PhoWhisper).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/phowhisper_small).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Setup
### 1. Install System-Level Dependencies
#### Linux
```bash
sudo apt install ffmpeg libportaudio2
```

#### Windows
```
winget install ffmpeg
```

### 2. Install the package
Install the package via pip:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.14 is supported.
pip install "qai-hub-models[phowhisper-small]"
```

### 3. Configure Qualcomm® AI Hub Workbench
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
python -m qai_hub_models.models.phowhisper_small.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment
To run the model on Qualcomm® devices, you must export the model for use with an edge runtime such as
TensorFlow Lite, ONNX Runtime, or Qualcomm AI Engine Direct. Use the following command to export the model:
```bash
python -m qai_hub_models.models.phowhisper_small.export --device "Samsung Galaxy S25 (Family)"
```
Additional options are documented with the `--help` option.

## License
* The license for the original implementation of PhoWhisper can be found
  [here](https://github.com/VinAIResearch/PhoWhisper/blob/main/LICENSE).

## References
* [PhoWhisper: Automatic Speech Recognition for Vietnamese](https://arxiv.org/abs/2406.02555)
* [Source Model Implementation](https://github.com/VinAIResearch/PhoWhisper)

## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
