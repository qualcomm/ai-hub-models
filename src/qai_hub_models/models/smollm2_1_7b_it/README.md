> [!WARNING]
> This model is not published. Use with caution; it may not meet performance/accuracy standards and may not support some runtimes or chipsets/devices. We do not provide support for unpublished models. If this model was previously published, use earlier releases.

# [SmolLM2-1.7B-Instruct: Compact language model capable of solving a wide range of tasks while being lightweight](https://aihub.qualcomm.com/models/smollm2_1_7b_it)

A 1.7B parameter instruction-tuned variant of SmolLM2, fine-tuned for conversational and instruction-following tasks, optimized for efficient on-device inference on Qualcomm Snapdragon platforms.

This is based on the implementation of SmolLM2-1.7B-Instruct found [here](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/smollm2_1_7b_it).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Quick Start

Use our lightweight command-line interface to inspect and download SmolLM2-1.7B-Instruct:

```bash
pip install qai_hub_models_cli # (the CLI is also available with the qai-hub-models package)

# Inspect the model and list the available download options
qai-hub-models info SmolLM2-1.7B-Instruct

# Print performance and accuracy metrics
qai-hub-models perf SmolLM2-1.7B-Instruct
qai-hub-models numerics SmolLM2-1.7B-Instruct

# Download a ready-to-deploy asset
qai-hub-models fetch SmolLM2-1.7B-Instruct --runtime geniex_qairt --precision w4a16
```
See the [CLI README](../../../../cli/README.md)
for the full list of commands and filters.

## Deploying SmolLM2-1.7B-Instruct on-device

Follow the [GenieX quickstart](https://geniex.aihub.qualcomm.com/en/get-started/quickstart) to install GenieX and deploy the model on a target device.

See the [LLM-on-Genie](https://github.com/qualcomm/ai-hub-apps/tree/main/tutorials/llm_on_genie) tutorial to run with the Genie runtime. Note: Genie support will be deprecated soon.


## Setup
### 1. Install the package
Install the base package, then use the `qai-hub-models` CLI to install this
recipe's dependencies:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.14 is supported.
pip install qai-hub-models
qai-hub-models install smollm2_1_7b_it
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
qai-hub-models demo smollm2_1_7b_it
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment
To run the model on Qualcomm® devices, you must export the model for use with an edge runtime such as
TensorFlow Lite, ONNX Runtime, or Qualcomm AI Engine Direct.
Export the pre-quantized model (published on AI Hub) for on-device deployment:
```bash
qai-hub-models export smollm2_1_7b_it --checkpoint DEFAULT_W4A16
```
`--checkpoint` also accepts `DEFAULT` (the model's default precision).

Optionally, quantize your own variant first and export the resulting checkpoint:
```bash
python -m qai_hub_models.models.smollm2_1_7b_it.quantize --precision w4a16 --output-dir ./quantized_checkpoint
qai-hub-models export smollm2_1_7b_it --checkpoint ./quantized_checkpoint
```
Additional options are documented with the `--help` option.

## License
* The license for the original implementation of SmolLM2-1.7B-Instruct can be found
  [here](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/blob/main/README.md).

## References
* [SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model](https://arxiv.org/abs/2502.02737)
* [Source Model Implementation](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)

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
