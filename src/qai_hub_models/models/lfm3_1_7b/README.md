# [LFM3-1.7B: Compact 1.7B hybrid language model from Liquid AI built for on-device deployment](https://aihub.qualcomm.com/models/lfm3_1_7b)

LFM3 is a family of hybrid language models from Liquid AI designed for fast, high-quality on-device deployment. The 1.7B model is tuned for instruction following, chat, and tool use.

This is based on the implementation of LFM3-1.7B found [here](https://huggingface.co/LiquidAI/LFM3-1.7B).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/lfm3_1_7b).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Quick Start

Use our lightweight command-line interface to inspect and download LFM3-1.7B:

```bash
pip install qai_hub_models_cli # (the CLI is also available with the qai-hub-models package)

# Inspect the model and list the available download options
qai-hub-models info LFM3-1.7B

# Print performance and accuracy metrics
qai-hub-models perf LFM3-1.7B
qai-hub-models numerics LFM3-1.7B
```
See the [CLI README](../../../../cli/README.md)
for the full list of commands and filters.

## Deploying LFM3-1.7B on-device

Follow the [GenieX quickstart](https://geniex.aihub.qualcomm.com/en/get-started/quickstart) to install GenieX and deploy the model on a target device.


## License
* The license for the original implementation of LFM3-1.7B can be found
  [here](https://www.liquid.ai/lfm-license).

## References
* [Introducing LFM3: Every Modality for Specialized Intelligence](https://www.liquid.ai/blog/lfm3)
* [Source Model Implementation](https://huggingface.co/LiquidAI/LFM3-1.7B)

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
