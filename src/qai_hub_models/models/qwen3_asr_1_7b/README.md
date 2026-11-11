# [Qwen3-ASR-1.7B: Multilingual 1.7B speech recognition model covering 30 languages and 22 Chinese dialects](https://aihub.qualcomm.com/models/qwen3_asr_1_7b)

Qwen3-ASR is a speech recognition model from the Qwen team, built on the audio understanding of Qwen3-Omni. It supports language identification and transcription for 30 languages and 22 Chinese dialects, and stays robust on noisy audio, singing voice, and songs with background music.

This is based on the implementation of Qwen3-ASR-1.7B found [here](https://huggingface.co/Qwen/Qwen3-ASR-1.7B).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/qwen3_asr_1_7b).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Quick Start

Use our lightweight command-line interface to inspect and download Qwen3-ASR-1.7B:

```bash
pip install qai_hub_models_cli # (the CLI is also available with the qai-hub-models package)

# Inspect the model and list the available download options
qai-hub-models info Qwen3-ASR-1.7B

# Print performance and accuracy metrics
qai-hub-models perf Qwen3-ASR-1.7B
qai-hub-models numerics Qwen3-ASR-1.7B
```
See the [CLI README](../../../../cli/README.md)
for the full list of commands and filters.

## Deploying Qwen3-ASR-1.7B on-device

Follow the [GenieX quickstart](https://geniex.aihub.qualcomm.com/en/get-started/quickstart) to install GenieX and deploy the model on a target device.


## License
* The license for the original implementation of Qwen3-ASR-1.7B can be found
  [here](https://www.apache.org/licenses/LICENSE-2.0).

## References
* [Qwen3-ASR Technical Report](https://arxiv.org/abs/2601.21337)
* [Source Model Implementation](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)

## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).

## Usage and Limitations

This model may not be used for or in connection with any of the following applications:

- Accessing essential private and public services and benefits;
- Administration of justice and democratic processes;
- Assessing or recognizing the emotional state of a person;
- Biometric and biometrics-based systems, including categorization of persons based on sensitive characteristics;
- Education and vocational training;
- Employment and workers management;
- Exploitation of the vulnerabilities of persons resulting in harmful behavior;
- General purpose social scoring;
- Law enforcement;
- Management and operation of critical infrastructure;
- Migration, asylum and border control management;
- Predictive policing;
- Real-time remote biometric identification in public spaces;
- Recommender systems of social media platforms;
- Scraping of facial images (from the internet or otherwise); and/or
- Subliminal manipulation
