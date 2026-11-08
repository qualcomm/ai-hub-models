# [Ministral-3-3B-Instruct-2512: Efficient 3B parameter instruction-tuned model from Mistral AI](https://aihub.qualcomm.com/models/ministral_3_3b_instruct_2512)

Ministral is a compact language model from Mistral AI designed for on-device deployment, offering strong instruction-following capabilities with low latency.

This is based on the implementation of Ministral-3-3B-Instruct-2512 found [here](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/ministral_3_3b_instruct_2512).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Quick Start

Use our lightweight command-line interface to inspect and download Ministral-3-3B-Instruct-2512:

```bash
pip install qai_hub_models_cli # (the CLI is also available with the qai-hub-models package)

# Inspect the model and list the available download options
qai-hub-models info Ministral-3-3B-Instruct-2512

# Print performance and accuracy metrics
qai-hub-models perf Ministral-3-3B-Instruct-2512
qai-hub-models numerics Ministral-3-3B-Instruct-2512
```
See the [CLI README](../../../../cli/README.md)
for the full list of commands and filters.

## Deploying Ministral-3-3B-Instruct-2512 on-device

Follow the [GenieX quickstart](https://geniex.aihub.qualcomm.com/en/get-started/quickstart) to install GenieX and deploy the model on a target device.


## License
* The license for the original implementation of Ministral-3-3B-Instruct-2512 can be found
  [here](https://www.apache.org/licenses/LICENSE-2.0).

## References
* [Ministral 3](https://arxiv.org/abs/2601.08584)
* [Source Model Implementation](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512)

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
