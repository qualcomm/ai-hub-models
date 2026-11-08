# [Granite-4.0-Micro: Compact language model from IBM optimized for enterprise tasks](https://aihub.qualcomm.com/models/granite_4_0_micro)

Granite 4.0 is a family of open language models from IBM designed for enterprise AI workloads including code generation, summarization, and retrieval-augmented generation.

This is based on the implementation of Granite-4.0-Micro found [here](https://huggingface.co/ibm-granite/granite-4.0-h-micro).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/granite_4_0_micro).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Quick Start

Use our lightweight command-line interface to inspect and download Granite-4.0-Micro:

```bash
pip install qai_hub_models_cli # (the CLI is also available with the qai-hub-models package)

# Inspect the model and list the available download options
qai-hub-models info Granite-4.0-Micro

# Print performance and accuracy metrics
qai-hub-models perf Granite-4.0-Micro
qai-hub-models numerics Granite-4.0-Micro
```
See the [CLI README](../../../../cli/README.md)
for the full list of commands and filters.

## Deploying Granite-4.0-Micro on-device

Follow the [GenieX quickstart](https://geniex.aihub.qualcomm.com/en/get-started/quickstart) to install GenieX and deploy the model on a target device.


## License
* The license for the original implementation of Granite-4.0-Micro can be found
  [here](https://github.com/ibm-granite/granite-4.0-language-models/blob/main/LICENSE).

## References
* [Granite 4.0](https://www.ibm.com/granite/docs/models/granite)
* [Source Model Implementation](https://huggingface.co/ibm-granite/granite-4.0-h-micro)

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
