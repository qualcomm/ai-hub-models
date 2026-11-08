# [Gemma-4-E2B-it: Lightweight multimodal model from Google DeepMind handling text and image input](https://aihub.qualcomm.com/models/gemma_4_e2b_it)

Gemma is a family of open models built by Google DeepMind. Gemma 4 models are multimodal, handling text and image input (with audio supported on small models) and generating text output. This release includes open-weights models in both pre-trained and instruction-tuned variants. Gemma 4 features a context window of up to 256K tokens and maintains multilingual support in over 140 languages.

This is based on the implementation of Gemma-4-E2B-it found [here](https://huggingface.co/google/gemma-4-E2B-it).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/gemma_4_e2b_it).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Quick Start

Use our lightweight command-line interface to inspect and download Gemma-4-E2B-it:

```bash
pip install qai_hub_models_cli # (the CLI is also available with the qai-hub-models package)

# Inspect the model and list the available download options
qai-hub-models info Gemma-4-E2B-it

# Print performance and accuracy metrics
qai-hub-models perf Gemma-4-E2B-it
qai-hub-models numerics Gemma-4-E2B-it
```
See the [CLI README](../../../../cli/README.md)
for the full list of commands and filters.

## Deploying Gemma-4-E2B-it on-device

Follow the [GenieX quickstart](https://geniex.aihub.qualcomm.com/en/get-started/quickstart) to install GenieX and deploy the model on a target device.


## License
* The license for the original implementation of Gemma-4-E2B-it can be found
  [here](https://ai.google.dev/gemma/apache_2).

## References
* [Gemma 4](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)
* [Source Model Implementation](https://huggingface.co/google/gemma-4-E2B-it)

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
