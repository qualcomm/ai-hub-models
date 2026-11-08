# [GPT-OSS-20B: State-of-the-art Mixture of Experts large language model with extended context length for text generation tasks](https://aihub.qualcomm.com/models/gpt_oss_20b)

GPT-OSS-20B is a 20.9B parameter Mixture of Experts (MoE) language model with 32 experts (4 active per token). It features an extended 131K context length with YARN rope scaling and uses a GPT-4o compatible tokenizer. The model is quantized to MXFP4 for efficient on-device deployment.

This is based on the implementation of GPT-OSS-20B found [here](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/gpt_oss_20b).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Quick Start

Use our lightweight command-line interface to inspect and download GPT-OSS-20B:

```bash
pip install qai_hub_models_cli # (the CLI is also available with the qai-hub-models package)

# Inspect the model and list the available download options
qai-hub-models info GPT-OSS-20B

# Print performance and accuracy metrics
qai-hub-models perf GPT-OSS-20B
qai-hub-models numerics GPT-OSS-20B
```
See the [CLI README](../../../../cli/README.md)
for the full list of commands and filters.

## Deploying GPT-OSS-20B on-device

Follow the [GenieX quickstart](https://geniex.aihub.qualcomm.com/en/get-started/quickstart) to install GenieX and deploy the model on a target device.


## License
* The license for the original implementation of GPT-OSS-20B can be found
  [here](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md).

## References
* [gpt-oss-120b & gpt-oss-20b Model Card](https://arxiv.org/abs/2508.10925)
* [Source Model Implementation](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF)

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
