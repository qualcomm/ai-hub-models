# [SmolVLM2-2.2B-Instruct: Multimodal 2.2B vision-language model capable of understanding text and images](https://aihub.qualcomm.com/models/smolvlm2_2_2b_instruct)

SmolVLM2 is a lightweight vision-language model from Hugging Face capable of understanding text and images for tasks such as visual question answering, image captioning, and optical character recognition.

This is based on the implementation of SmolVLM2-2.2B-Instruct found [here](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/smolvlm2_2_2b_instruct).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Quick Start

Use our lightweight command-line interface to inspect and download SmolVLM2-2.2B-Instruct:

```bash
pip install qai_hub_models_cli # (the CLI is also available with the qai-hub-models package)

# Inspect the model and list the available download options
qai-hub-models info SmolVLM2-2.2B-Instruct

# Print performance and accuracy metrics
qai-hub-models perf SmolVLM2-2.2B-Instruct
qai-hub-models numerics SmolVLM2-2.2B-Instruct
```
See the [CLI README](../../../../cli/README.md)
for the full list of commands and filters.

## Deploying SmolVLM2-2.2B-Instruct on-device

Follow the [GenieX quickstart](https://geniex.aihub.qualcomm.com/en/get-started/quickstart) to install GenieX and deploy the model on a target device.


## License
* The license for the original implementation of SmolVLM2-2.2B-Instruct can be found
  [here](https://www.apache.org/licenses/LICENSE-2.0).

## References
* [SmolVLM: Redefining small and efficient multimodal models](https://arxiv.org/abs/2504.05299)
* [Source Model Implementation](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct)

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
