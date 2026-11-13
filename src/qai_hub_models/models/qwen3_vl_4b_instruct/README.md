# [Qwen3-VL-4B-Instruct: Multimodal 4B vision-language model with enhanced visual reasoning capabilities](https://aihub.qualcomm.com/models/qwen3_vl_4b_instruct)

Qwen3-VL is a vision-language model from Alibaba Cloud capable of understanding both text and images for multimodal reasoning tasks such as visual question answering and image captioning.

This is based on the implementation of Qwen3-VL-4B-Instruct found [here](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/qwen3_vl_4b_instruct).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Quick Start

Use our lightweight command-line interface to inspect and download Qwen3-VL-4B-Instruct:

```bash
pip install qai_hub_models_cli # (the CLI is also available with the qai-hub-models package)

# Inspect the model and list the available download options
qai-hub-models info Qwen3-VL-4B-Instruct

# Print performance and accuracy metrics
qai-hub-models perf Qwen3-VL-4B-Instruct
qai-hub-models numerics Qwen3-VL-4B-Instruct

# Download a ready-to-deploy asset
qai-hub-models fetch Qwen3-VL-4B-Instruct --runtime geniex_qairt --precision w4a16
```
See the [CLI README](../../../../cli/README.md)
for the full list of commands and filters.

## Deploying Qwen3-VL-4B-Instruct on-device

Follow the [GenieX quickstart](https://geniex.aihub.qualcomm.com/en/get-started/quickstart) to install GenieX and deploy the model on a target device.

See the [LLM-on-Genie](https://github.com/qualcomm/ai-hub-apps/tree/main/tutorials/llm_on_genie) tutorial to run with the Genie runtime. Note: Genie support will be deprecated soon.


## Setup
### 1. Install the package
Install the base package, then use the `qai-hub-models` CLI to install this
recipe's dependencies:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.13 is supported.
pip install qai-hub-models
qai-hub-models install qwen3_vl_4b_instruct
```
For qwen3_vl_4b_instruct, some additional functionality can be faster or is available
only with a GPU on the host machine.

- 🟢 Exporting the model for on-device deployment (GPU not required)
- 🟡 Running the demo (GPU recommended for speed, but not required)
- 🟡 Running evaluation (GPU recommended for speed, but not required)
- 🔴 Quantizing the model (GPU required)

If you are quantizing your own variant of qwen3_vl_4b_instruct, a dedicated CUDA enabled
GPU (40 GB VRAM for 3B models to 80 GB VRAM for 8B models) is recommended. A GPU
can also increase the speed of evaluation and demo of your quantized model
significantly but is not strictly required. The CLI auto-detects CUDA and installs
the GPU-flavored dependencies (e.g. the AIMET ONNX wheel) when available.

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
qai-hub-models demo qwen3_vl_4b_instruct
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
qai-hub-models export qwen3_vl_4b_instruct --checkpoint DEFAULT_W4A16
```
`--checkpoint` also accepts `DEFAULT` (the model's default precision) or any of `DEFAULT_W4A16`, `DEFAULT_Q4_0`.

Optionally, quantize your own variant first and export the resulting checkpoint:
```bash
python -m qai_hub_models.models.qwen3_vl_4b_instruct.quantize --precision w4a16 --output-dir ./quantized_checkpoint
qai-hub-models export qwen3_vl_4b_instruct --checkpoint ./quantized_checkpoint
```
Additional options are documented with the `--help` option.

## License
* The license for the original implementation of Qwen3-VL-4B-Instruct can be found
  [here](https://www.apache.org/licenses/LICENSE-2.0).

## References
* [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
* [Source Model Implementation](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)

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
