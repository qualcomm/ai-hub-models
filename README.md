# [Qualcomm® AI Hub Models](https://aihub.qualcomm.com/)

[![Release](https://img.shields.io/github/v/release/quic/ai-hub-models)](https://github.com/quic/ai-hub-models/releases/latest)
[![Tag](https://img.shields.io/github/v/tag/quic/ai-hub-models)](https://github.com/quic/ai-hub-models/releases/latest)
[![PyPi](https://img.shields.io/pypi/v/qai-hub-models)](https://pypi.org/project/qai-hub-models/)
![Python 3.10, 3.11, 3.12, 3.13](https://img.shields.io/badge/python-3.10%20(Recommended)%2C%203.11%2C%203.12%2C%203.13-yellow)

The Qualcomm® AI Hub Models are a collection of
state-of-the-art machine learning models optimized for deployment on Qualcomm® devices.

* [List of Models by Category](#model-directory)
* [On-Device Performance Data](https://aihub.qualcomm.com/models)
* [Device-Native Sample Apps](https://github.com/quic/ai-hub-apps)

See supported: [On-Device Runtimes](#on-device-runtimes), [Hardware Targets & Precision](#device-hardware--precision), [Chipsets](#chipsets), [Devices](#devices)

&nbsp;

## Setup

### 1. Install Python Package

The package is available via pip:

```shell
# NOTE for Snapdragon X Elite users:
# Only AMDx64 (64-bit) Python is supported on Windows.
# Installation will fail when using Windows ARM64 Python.

pip install qai_hub_models
```

Some models (e.g. [YOLOv7](https://github.com/quic/ai-hub-models/tree/main/qai_hub_models/models/yolov7)) require
additional dependencies. View the model README (at [qai_hub_models/models/model_id](https://github.com/quic/ai-hub-models/tree/main/qai_hub_models/models/yolov7)) for installation instructions.

&nbsp;

### 2. Configure AI Hub Workbench Access

Many features of AI Hub Models _(such as model compilation, on-device profiling, etc.)_ require access to Qualcomm® AI Hub Workbench:

-  [Create a Qualcomm® ID](https://myaccount.qualcomm.com/signup), and use it to [login to Qualcomm® AI Hub Workbench](https://workbench.aihub.qualcomm.com/).
-  Configure your [API token](https://workbench.aihub.qualcomm.com/account/): `qai-hub configure --api_token API_TOKEN`

&nbsp;

## Getting Started

### Export and Run A Model on a Physical Device

All [models in our directory](#model-directory) can be compiled and profiled on a hosted
Qualcomm® device:

```shell
pip install "qai_hub_models[yolov7]"

python -m qai_hub_models.models.yolov7.export [--target-runtime ...] [--device ...] [--help]
```

_Using Qualcomm® AI Hub Workbench_, the export script will:

1. **Compile** the model for the chosen device and target runtime (see: [Compiling Models on AI Hub Workbench](https://workbench.aihub.qualcomm.com/docs/hub/compile_examples.html)).
2. If applicable, **Quantize** the model (see: [Quantization on AI Hub Workbench](https://workbench.aihub.qualcomm.com/docs/hub/quantize_examples.html))
3. **Profile** the compiled model on a real device in the cloud (see: [Profiling Models on AI Hub Workbench](https://workbench.aihub.qualcomm.com/docs/hub/profile_examples.html)).
4. **Run inference** with a sample input data on a real device in the cloud, and compare on-device model output with PyTorch output (see: [Running Inference on AI Hub Workbench](https://workbench.aihub.qualcomm.com/docs/hub/inference_examples.html))
5. **Download** the compiled model to disk.

&nbsp;

### End-To-End Model Demos

Most [models in our directory](#model-directory) contain CLI demos that run the model _end-to-end_:

```shell
pip install "qai_hub_models[yolov7]"
# Predict and draw bounding boxes on the provided image
python -m qai_hub_models.models.yolov7.demo [--image ...] [--eval-mode {fp,on-device}] [--help]
```

_End-to-end_ demos:
1. **Preprocess** human-readable input into model input
2. Run **model inference**
3. **Postprocess** model output to a human-readable format

**Many end-to-end demos use AI Hub Workbench to run inference on a real cloud-hosted device** _(with `--eval-mode on-device`)_. All end-to-end demos can also run locally via PyTorch (with `--eval-mode fp`).

&nbsp;

### Sample Applications

**Native** applications that can run our models (with pre- and post-processing) on physical devices are published in the [AI Hub Apps repository](https://github.com/quic/ai-hub-apps/).

**Python** applications are defined for all models [(from qai_hub_models.models.\<model_name> import App)](https://github.com/quic/ai-hub-models/blob/main/qai_hub_models/models/yolov7/app.py). These apps wrap model inference with pre- and post-processing steps written using torch & numpy. **These apps are optimized to be an easy-to-follow example, rather than to minimize prediction time.**

&nbsp;

## Model Support Data

### On-Device Runtimes

| Runtime | Supported OS |
| -- | -- |
| [Qualcomm AI Engine Direct](https://www.qualcomm.com/developer/artificial-intelligence#overview) | Android, Linux, Windows
| [LiteRT (TensorFlow Lite)](https://www.tensorflow.org/lite) | Android, Linux
| [ONNX](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html) | Android, Linux, Windows

### Device Hardware & Precision

| Device Compute Unit | Supported Precision |
| -- | -- |
| CPU | FP32, INT16, INT8
| GPU | FP32, FP16
| NPU (includes [Hexagon DSP](https://developer.qualcomm.com/software/hexagon-dsp-sdk/dsp-processor), [HTP](https://developer.qualcomm.com/hardware/qualcomm-innovators-development-kit/ai-resources-overview/ai-hardware-cores-accelerators)) | FP16*, INT16, INT8

*Some older chipsets do not support fp16 inference on their NPU.

### Chipsets
* Snapdragon [8 Elite](https://www.qualcomm.com/products/mobile/snapdragon/smartphones/snapdragon-8-series-mobile-platforms/snapdragon-8-elite-mobile-platform), [8 Gen 3](https://www.qualcomm.com/products/mobile/snapdragon/smartphones/snapdragon-8-series-mobile-platforms/snapdragon-8-gen-3-mobile-platform), [8 Gen 2](https://www.qualcomm.com/products/mobile/snapdragon/smartphones/snapdragon-8-series-mobile-platforms/snapdragon-8-gen-2-mobile-platform), and [8 Gen 1](https://www.qualcomm.com/products/mobile/snapdragon/smartphones/snapdragon-8-series-mobile-platforms/snapdragon-8-gen-1-mobile-platform) Mobile Platforms
* [Snapdragon X Elite](https://www.qualcomm.com/products/mobile/snapdragon/pcs-and-tablets/snapdragon-x-elite) Compute Platform
* SA8255P, SA8295P, SA8650P, and SA8775P Automotive Platforms
* [QCS 6490](https://www.qualcomm.com/products/internet-of-things/industrial/building-enterprise/qcs6490),  [QCS 8250](https://www.qualcomm.com/products/internet-of-things/consumer/cameras/qcs8250), and [QCS 8550](https://www.qualcomm.com/products/technology/processors/qcs8550) IoT Platforms
* QCS8450 XR Platform

and many more.

### Devices
* Samsung Galaxy S21, S22, S23, and S24 Series
* Xiaomi 12 and 13
* Snapdragon X Elite CRD (Compute Reference Device)
* Qualcomm RB3 Gen 2, RB5

and many more.

&nbsp;

## Model Directory

### Computer Vision

| Model | README |
| -- | -- |
| | |
| **Image Classification**
| [Beit](https://aihub.qualcomm.com/models/beit) | [qai_hub_models.models.beit](qai_hub_models/models/beit/README.md) |
| [ConvNext-Base](https://aihub.qualcomm.com/models/convnext_base) | [qai_hub_models.models.convnext_base](qai_hub_models/models/convnext_base/README.md) |
| [ConvNext-Tiny](https://aihub.qualcomm.com/models/convnext_tiny) | [qai_hub_models.models.convnext_tiny](qai_hub_models/models/convnext_tiny/README.md) |
| [DLA-102-X](https://aihub.qualcomm.com/models/dla102x) | [qai_hub_models.models.dla102x](qai_hub_models/models/dla102x/README.md) |
| [DenseNet-121](https://aihub.qualcomm.com/models/densenet121) | [qai_hub_models.models.densenet121](qai_hub_models/models/densenet121/README.md) |
| [EfficientFormer](https://aihub.qualcomm.com/models/efficientformer) | [qai_hub_models.models.efficientformer](qai_hub_models/models/efficientformer/README.md) |
| [EfficientNet-B0](https://aihub.qualcomm.com/models/efficientnet_b0) | [qai_hub_models.models.efficientnet_b0](qai_hub_models/models/efficientnet_b0/README.md) |
| [EfficientNet-B4](https://aihub.qualcomm.com/models/efficientnet_b4) | [qai_hub_models.models.efficientnet_b4](qai_hub_models/models/efficientnet_b4/README.md) |
| [EfficientNet-V2-s](https://aihub.qualcomm.com/models/efficientnet_v2_s) | [qai_hub_models.models.efficientnet_v2_s](qai_hub_models/models/efficientnet_v2_s/README.md) |
| [EfficientViT-b2-cls](https://aihub.qualcomm.com/models/efficientvit_b2_cls) | [qai_hub_models.models.efficientvit_b2_cls](qai_hub_models/models/efficientvit_b2_cls/README.md) |
| [EfficientViT-l2-cls](https://aihub.qualcomm.com/models/efficientvit_l2_cls) | [qai_hub_models.models.efficientvit_l2_cls](qai_hub_models/models/efficientvit_l2_cls/README.md) |
| [GPUNet](https://aihub.qualcomm.com/models/gpunet) | [qai_hub_models.models.gpunet](qai_hub_models/models/gpunet/README.md) |
| [GoogLeNet](https://aihub.qualcomm.com/models/googlenet) | [qai_hub_models.models.googlenet](qai_hub_models/models/googlenet/README.md) |
| [Inception-v3](https://aihub.qualcomm.com/models/inception_v3) | [qai_hub_models.models.inception_v3](qai_hub_models/models/inception_v3/README.md) |
| [InternImage](https://aihub.qualcomm.com/models/internimage) | [qai_hub_models.models.internimage](qai_hub_models/models/internimage/README.md) |
| [LeViT](https://aihub.qualcomm.com/models/levit) | [qai_hub_models.models.levit](qai_hub_models/models/levit/README.md) |
| [MNASNet05](https://aihub.qualcomm.com/models/mnasnet05) | [qai_hub_models.models.mnasnet05](qai_hub_models/models/mnasnet05/README.md) |
| [Mobile-VIT](https://aihub.qualcomm.com/models/mobile_vit) | [qai_hub_models.models.mobile_vit](qai_hub_models/models/mobile_vit/README.md) |
| [MobileNet-v2](https://aihub.qualcomm.com/models/mobilenet_v2) | [qai_hub_models.models.mobilenet_v2](qai_hub_models/models/mobilenet_v2/README.md) |
| [MobileNet-v3-Large](https://aihub.qualcomm.com/models/mobilenet_v3_large) | [qai_hub_models.models.mobilenet_v3_large](qai_hub_models/models/mobilenet_v3_large/README.md) |
| [MobileNet-v3-Small](https://aihub.qualcomm.com/models/mobilenet_v3_small) | [qai_hub_models.models.mobilenet_v3_small](qai_hub_models/models/mobilenet_v3_small/README.md) |
| [NASNet](https://aihub.qualcomm.com/models/nasnet) | [qai_hub_models.models.nasnet](qai_hub_models/models/nasnet/README.md) |
| [RegNet](https://aihub.qualcomm.com/models/regnet) | [qai_hub_models.models.regnet](qai_hub_models/models/regnet/README.md) |
| [RegNet-Y-800MF](https://aihub.qualcomm.com/models/regnet_y_800mf) | [qai_hub_models.models.regnet_y_800mf](qai_hub_models/models/regnet_y_800mf/README.md) |
| [ResNeXt101](https://aihub.qualcomm.com/models/resnext101) | [qai_hub_models.models.resnext101](qai_hub_models/models/resnext101/README.md) |
| [ResNeXt50](https://aihub.qualcomm.com/models/resnext50) | [qai_hub_models.models.resnext50](qai_hub_models/models/resnext50/README.md) |
| [ResNet101](https://aihub.qualcomm.com/models/resnet101) | [qai_hub_models.models.resnet101](qai_hub_models/models/resnet101/README.md) |
| [ResNet18](https://aihub.qualcomm.com/models/resnet18) | [qai_hub_models.models.resnet18](qai_hub_models/models/resnet18/README.md) |
| [ResNet50](https://aihub.qualcomm.com/models/resnet50) | [qai_hub_models.models.resnet50](qai_hub_models/models/resnet50/README.md) |
| [Sequencer2D](https://aihub.qualcomm.com/models/sequencer2d) | [qai_hub_models.models.sequencer2d](qai_hub_models/models/sequencer2d/README.md) |
| [Shufflenet-v2](https://aihub.qualcomm.com/models/shufflenet_v2) | [qai_hub_models.models.shufflenet_v2](qai_hub_models/models/shufflenet_v2/README.md) |
| [SqueezeNet-1.1](https://aihub.qualcomm.com/models/squeezenet1_1) | [qai_hub_models.models.squeezenet1_1](qai_hub_models/models/squeezenet1_1/README.md) |
| [VIT](https://aihub.qualcomm.com/models/vit) | [qai_hub_models.models.vit](qai_hub_models/models/vit/README.md) |
| [WideResNet50](https://aihub.qualcomm.com/models/wideresnet50) | [qai_hub_models.models.wideresnet50](qai_hub_models/models/wideresnet50/README.md) |
| | |
| **Image Editing**
| [AOT-GAN](https://aihub.qualcomm.com/models/aotgan) | [qai_hub_models.models.aotgan](qai_hub_models/models/aotgan/README.md) |
| [DDColor](https://aihub.qualcomm.com/models/ddcolor) | [qai_hub_models.models.ddcolor](qai_hub_models/models/ddcolor/README.md) |
| [LaMa-Dilated](https://aihub.qualcomm.com/models/lama_dilated) | [qai_hub_models.models.lama_dilated](qai_hub_models/models/lama_dilated/README.md) |
| | |
| **Super Resolution**
| [Real-ESRGAN-x4plus](https://aihub.qualcomm.com/models/real_esrgan_x4plus) | [qai_hub_models.models.real_esrgan_x4plus](qai_hub_models/models/real_esrgan_x4plus/README.md) |
| [XLSR](https://aihub.qualcomm.com/models/xlsr) | [qai_hub_models.models.xlsr](qai_hub_models/models/xlsr/README.md) |
| | |
| **Semantic Segmentation**
| [BGNet](https://aihub.qualcomm.com/models/bgnet) | [qai_hub_models.models.bgnet](qai_hub_models/models/bgnet/README.md) |
| [BiseNet](https://aihub.qualcomm.com/models/bisenet) | [qai_hub_models.models.bisenet](qai_hub_models/models/bisenet/README.md) |
| [DDRNet23-Slim](https://aihub.qualcomm.com/models/ddrnet23_slim) | [qai_hub_models.models.ddrnet23_slim](qai_hub_models/models/ddrnet23_slim/README.md) |
| [DeepLabV3-Plus-MobileNet](https://aihub.qualcomm.com/models/deeplabv3_plus_mobilenet) | [qai_hub_models.models.deeplabv3_plus_mobilenet](qai_hub_models/models/deeplabv3_plus_mobilenet/README.md) |
| [DeepLabXception](https://aihub.qualcomm.com/models/deeplab_xception) | [qai_hub_models.models.deeplab_xception](qai_hub_models/models/deeplab_xception/README.md) |
| [EdgeTAM](https://aihub.qualcomm.com/models/edgetam) | [qai_hub_models.models.edgetam](qai_hub_models/models/edgetam/README.md) |
| [FCN-ResNet50](https://aihub.qualcomm.com/models/fcn_resnet50) | [qai_hub_models.models.fcn_resnet50](qai_hub_models/models/fcn_resnet50/README.md) |
| [FFNet-122NS-LowRes](https://aihub.qualcomm.com/models/ffnet_122ns_lowres) | [qai_hub_models.models.ffnet_122ns_lowres](qai_hub_models/models/ffnet_122ns_lowres/README.md) |
| [FFNet-40S](https://aihub.qualcomm.com/models/ffnet_40s) | [qai_hub_models.models.ffnet_40s](qai_hub_models/models/ffnet_40s/README.md) |
| [FFNet-54S](https://aihub.qualcomm.com/models/ffnet_54s) | [qai_hub_models.models.ffnet_54s](qai_hub_models/models/ffnet_54s/README.md) |
| [FFNet-78S](https://aihub.qualcomm.com/models/ffnet_78s) | [qai_hub_models.models.ffnet_78s](qai_hub_models/models/ffnet_78s/README.md) |
| [FFNet-78S-LowRes](https://aihub.qualcomm.com/models/ffnet_78s_lowres) | [qai_hub_models.models.ffnet_78s_lowres](qai_hub_models/models/ffnet_78s_lowres/README.md) |
| [FastSam-S](https://aihub.qualcomm.com/models/fastsam_s) | [qai_hub_models.models.fastsam_s](qai_hub_models/models/fastsam_s/README.md) |
| [FastSam-X](https://aihub.qualcomm.com/models/fastsam_x) | [qai_hub_models.models.fastsam_x](qai_hub_models/models/fastsam_x/README.md) |
| [HRNet-W48-OCR](https://aihub.qualcomm.com/models/hrnet_w48_ocr) | [qai_hub_models.models.hrnet_w48_ocr](qai_hub_models/models/hrnet_w48_ocr/README.md) |
| [MaskRCNN](qai_hub_models/models/maskrcnn/README.md) | [qai_hub_models.models.maskrcnn](qai_hub_models/models/maskrcnn/README.md) |
| [MediaPipe-Selfie-Segmentation](https://aihub.qualcomm.com/models/mediapipe_selfie) | [qai_hub_models.models.mediapipe_selfie](qai_hub_models/models/mediapipe_selfie/README.md) |
| [MobileSam](https://aihub.qualcomm.com/models/mobilesam) | [qai_hub_models.models.mobilesam](qai_hub_models/models/mobilesam/README.md) |
| [PSPNet](https://aihub.qualcomm.com/models/pspnet) | [qai_hub_models.models.pspnet](qai_hub_models/models/pspnet/README.md) |
| [PidNet](https://aihub.qualcomm.com/models/pidnet) | [qai_hub_models.models.pidnet](qai_hub_models/models/pidnet/README.md) |
| [SINet](https://aihub.qualcomm.com/models/sinet) | [qai_hub_models.models.sinet](qai_hub_models/models/sinet/README.md) |
| [SalsaNext](https://aihub.qualcomm.com/models/salsanext) | [qai_hub_models.models.salsanext](qai_hub_models/models/salsanext/README.md) |
| [Segformer-Base](https://aihub.qualcomm.com/models/segformer_base) | [qai_hub_models.models.segformer_base](qai_hub_models/models/segformer_base/README.md) |
| [Segment-Anything-Model-2](https://aihub.qualcomm.com/models/sam2) | [qai_hub_models.models.sam2](qai_hub_models/models/sam2/README.md) |
| [Unet-Segmentation](https://aihub.qualcomm.com/models/unet_segmentation) | [qai_hub_models.models.unet_segmentation](qai_hub_models/models/unet_segmentation/README.md) |
| [YOLOv11-Segmentation](https://aihub.qualcomm.com/models/yolov11_seg) | [qai_hub_models.models.yolov11_seg](qai_hub_models/models/yolov11_seg/README.md) |
| [YOLOv8-Segmentation](https://aihub.qualcomm.com/models/yolov8_seg) | [qai_hub_models.models.yolov8_seg](qai_hub_models/models/yolov8_seg/README.md) |
| | |
| **Video Classification**
| [ResNet-2Plus1D](https://aihub.qualcomm.com/models/resnet_2plus1d) | [qai_hub_models.models.resnet_2plus1d](qai_hub_models/models/resnet_2plus1d/README.md) |
| [ResNet-3D](https://aihub.qualcomm.com/models/resnet_3d) | [qai_hub_models.models.resnet_3d](qai_hub_models/models/resnet_3d/README.md) |
| [ResNet-Mixed-Convolution](https://aihub.qualcomm.com/models/resnet_mixed) | [qai_hub_models.models.resnet_mixed](qai_hub_models/models/resnet_mixed/README.md) |
| [Video-MAE](https://aihub.qualcomm.com/models/video_mae) | [qai_hub_models.models.video_mae](qai_hub_models/models/video_mae/README.md) |
| | |
| **Video Generation**
| [First-Order-Motion-Model](https://aihub.qualcomm.com/models/fomm) | [qai_hub_models.models.fomm](qai_hub_models/models/fomm/README.md) |
| | |
| **Video Object Tracking**
| [Track-Anything](https://aihub.qualcomm.com/models/track_anything) | [qai_hub_models.models.track_anything](qai_hub_models/models/track_anything/README.md) |
| | |
| **Object Detection**
| [3D-Deep-BOX](https://aihub.qualcomm.com/models/deepbox) | [qai_hub_models.models.deepbox](qai_hub_models/models/deepbox/README.md) |
| [CavaFace](https://aihub.qualcomm.com/models/cavaface) | [qai_hub_models.models.cavaface](qai_hub_models/models/cavaface/README.md) |
| [CenterNet-2D](https://aihub.qualcomm.com/models/centernet_2d) | [qai_hub_models.models.centernet_2d](qai_hub_models/models/centernet_2d/README.md) |
| [Conditional-DETR-ResNet50](https://aihub.qualcomm.com/models/conditional_detr_resnet50) | [qai_hub_models.models.conditional_detr_resnet50](qai_hub_models/models/conditional_detr_resnet50/README.md) |
| [DETR-ResNet101](https://aihub.qualcomm.com/models/detr_resnet101) | [qai_hub_models.models.detr_resnet101](qai_hub_models/models/detr_resnet101/README.md) |
| [DETR-ResNet101-DC5](https://aihub.qualcomm.com/models/detr_resnet101_dc5) | [qai_hub_models.models.detr_resnet101_dc5](qai_hub_models/models/detr_resnet101_dc5/README.md) |
| [DETR-ResNet50](https://aihub.qualcomm.com/models/detr_resnet50) | [qai_hub_models.models.detr_resnet50](qai_hub_models/models/detr_resnet50/README.md) |
| [DETR-ResNet50-DC5](https://aihub.qualcomm.com/models/detr_resnet50_dc5) | [qai_hub_models.models.detr_resnet50_dc5](qai_hub_models/models/detr_resnet50_dc5/README.md) |
| [DeformableDETR](https://aihub.qualcomm.com/models/deformable_detr) | [qai_hub_models.models.deformable_detr](qai_hub_models/models/deformable_detr/README.md) |
| [Detectron2-Detection](https://aihub.qualcomm.com/models/detectron2_detection) | [qai_hub_models.models.detectron2_detection](qai_hub_models/models/detectron2_detection/README.md) |
| [Facial-Attribute-Detection](https://aihub.qualcomm.com/models/face_attrib_net) | [qai_hub_models.models.face_attrib_net](qai_hub_models/models/face_attrib_net/README.md) |
| [HRNetFace](https://aihub.qualcomm.com/models/hrnet_face) | [qai_hub_models.models.hrnet_face](qai_hub_models/models/hrnet_face/README.md) |
| [Lightweight-Face-Detection](https://aihub.qualcomm.com/models/face_det_lite) | [qai_hub_models.models.face_det_lite](qai_hub_models/models/face_det_lite/README.md) |
| [MediaPipe-Face-Detection](https://aihub.qualcomm.com/models/mediapipe_face) | [qai_hub_models.models.mediapipe_face](qai_hub_models/models/mediapipe_face/README.md) |
| [MediaPipe-Hand-Detection](https://aihub.qualcomm.com/models/mediapipe_hand) | [qai_hub_models.models.mediapipe_hand](qai_hub_models/models/mediapipe_hand/README.md) |
| [MediaPipe-Hand-Gesture-Recognition](https://aihub.qualcomm.com/models/mediapipe_hand_gesture) | [qai_hub_models.models.mediapipe_hand_gesture](qai_hub_models/models/mediapipe_hand_gesture/README.md) |
| [PPE-Detection](https://aihub.qualcomm.com/models/gear_guard_net) | [qai_hub_models.models.gear_guard_net](qai_hub_models/models/gear_guard_net/README.md) |
| [Person-Foot-Detection](https://aihub.qualcomm.com/models/foot_track_net) | [qai_hub_models.models.foot_track_net](qai_hub_models/models/foot_track_net/README.md) |
| [RTMDet](https://aihub.qualcomm.com/models/rtmdet) | [qai_hub_models.models.rtmdet](qai_hub_models/models/rtmdet/README.md) |
| [YOLOv10-Detection](https://aihub.qualcomm.com/models/yolov10_det) | [qai_hub_models.models.yolov10_det](qai_hub_models/models/yolov10_det/README.md) |
| [YOLOv11-Detection](https://aihub.qualcomm.com/models/yolov11_det) | [qai_hub_models.models.yolov11_det](qai_hub_models/models/yolov11_det/README.md) |
| [YOLOv8-Detection](https://aihub.qualcomm.com/models/yolov8_det) | [qai_hub_models.models.yolov8_det](qai_hub_models/models/yolov8_det/README.md) |
| [Yolo-R](https://aihub.qualcomm.com/models/yolor) | [qai_hub_models.models.yolor](qai_hub_models/models/yolor/README.md) |
| [Yolo-X](https://aihub.qualcomm.com/models/yolox) | [qai_hub_models.models.yolox](qai_hub_models/models/yolox/README.md) |
| [Yolo-v3](https://aihub.qualcomm.com/models/yolov3) | [qai_hub_models.models.yolov3](qai_hub_models/models/yolov3/README.md) |
| [Yolo-v5](https://aihub.qualcomm.com/models/yolov5) | [qai_hub_models.models.yolov5](qai_hub_models/models/yolov5/README.md) |
| [Yolo-v6](https://aihub.qualcomm.com/models/yolov6) | [qai_hub_models.models.yolov6](qai_hub_models/models/yolov6/README.md) |
| [Yolo-v7](https://aihub.qualcomm.com/models/yolov7) | [qai_hub_models.models.yolov7](qai_hub_models/models/yolov7/README.md) |
| | |
| **Pose Estimation**
| [CenterNet-Pose](https://aihub.qualcomm.com/models/centernet_pose) | [qai_hub_models.models.centernet_pose](qai_hub_models/models/centernet_pose/README.md) |
| [Facial-Landmark-Detection](https://aihub.qualcomm.com/models/facemap_3dmm) | [qai_hub_models.models.facemap_3dmm](qai_hub_models/models/facemap_3dmm/README.md) |
| [HRNetPose](https://aihub.qualcomm.com/models/hrnet_pose) | [qai_hub_models.models.hrnet_pose](qai_hub_models/models/hrnet_pose/README.md) |
| [LiteHRNet](https://aihub.qualcomm.com/models/litehrnet) | [qai_hub_models.models.litehrnet](qai_hub_models/models/litehrnet/README.md) |
| [MediaPipe-Pose-Estimation](https://aihub.qualcomm.com/models/mediapipe_pose) | [qai_hub_models.models.mediapipe_pose](qai_hub_models/models/mediapipe_pose/README.md) |
| [Posenet-Mobilenet](https://aihub.qualcomm.com/models/posenet_mobilenet) | [qai_hub_models.models.posenet_mobilenet](qai_hub_models/models/posenet_mobilenet/README.md) |
| [RTMPose-Body2d](https://aihub.qualcomm.com/models/rtmpose_body2d) | [qai_hub_models.models.rtmpose_body2d](qai_hub_models/models/rtmpose_body2d/README.md) |
| | |
| **Gaze Estimation**
| [EyeGaze](https://aihub.qualcomm.com/models/eyegaze) | [qai_hub_models.models.eyegaze](qai_hub_models/models/eyegaze/README.md) |
| | |
| **Depth Estimation**
| [Depth-Anything](https://aihub.qualcomm.com/models/depth_anything) | [qai_hub_models.models.depth_anything](qai_hub_models/models/depth_anything/README.md) |
| [Depth-Anything-V2](https://aihub.qualcomm.com/models/depth_anything_v2) | [qai_hub_models.models.depth_anything_v2](qai_hub_models/models/depth_anything_v2/README.md) |
| [Depth-Anything-V3](https://aihub.qualcomm.com/models/depth_anything_v3) | [qai_hub_models.models.depth_anything_v3](qai_hub_models/models/depth_anything_v3/README.md) |
| [Midas-V2](https://aihub.qualcomm.com/models/midas) | [qai_hub_models.models.midas](qai_hub_models/models/midas/README.md) |
| | |
| **Driver Assistance**
| [BEVDet](https://aihub.qualcomm.com/models/bevdet) | [qai_hub_models.models.bevdet](qai_hub_models/models/bevdet/README.md) |
| [BEVFusion](https://aihub.qualcomm.com/models/bevfusion_det) | [qai_hub_models.models.bevfusion_det](qai_hub_models/models/bevfusion_det/README.md) |
| [CVT](https://aihub.qualcomm.com/models/cvt) | [qai_hub_models.models.cvt](qai_hub_models/models/cvt/README.md) |
| [CenterNet-3D](https://aihub.qualcomm.com/models/centernet_3d) | [qai_hub_models.models.centernet_3d](qai_hub_models/models/centernet_3d/README.md) |
| [CenterPoint](https://aihub.qualcomm.com/models/centerpoint) | [qai_hub_models.models.centerpoint](qai_hub_models/models/centerpoint/README.md) |
| [GKT](https://aihub.qualcomm.com/models/gkt) | [qai_hub_models.models.gkt](qai_hub_models/models/gkt/README.md) |
| [StateTransformer](https://aihub.qualcomm.com/models/statetransformer) | [qai_hub_models.models.statetransformer](qai_hub_models/models/statetransformer/README.md) |
| | |
| **Robotics**
| [ACT](https://aihub.qualcomm.com/models/act) | [qai_hub_models.models.act](qai_hub_models/models/act/README.md) |

### Multimodal

| Model | README |
| -- | -- |
| | |
| [EasyOCR](https://aihub.qualcomm.com/models/easyocr) | [qai_hub_models.models.easyocr](qai_hub_models/models/easyocr/README.md) |
| [Nomic-Embed-Text](https://aihub.qualcomm.com/models/nomic_embed_text) | [qai_hub_models.models.nomic_embed_text](qai_hub_models/models/nomic_embed_text/README.md) |
| [OpenAI-Clip](https://aihub.qualcomm.com/models/openai_clip) | [qai_hub_models.models.openai_clip](qai_hub_models/models/openai_clip/README.md) |
| [OpusMT-En-Es](https://aihub.qualcomm.com/models/opus_mt_en_es) | [qai_hub_models.models.opus_mt_en_es](qai_hub_models/models/opus_mt_en_es/README.md) |
| [OpusMT-En-Zh](https://aihub.qualcomm.com/models/opus_mt_en_zh) | [qai_hub_models.models.opus_mt_en_zh](qai_hub_models/models/opus_mt_en_zh/README.md) |
| [OpusMT-Es-En](https://aihub.qualcomm.com/models/opus_mt_es_en) | [qai_hub_models.models.opus_mt_es_en](qai_hub_models/models/opus_mt_es_en/README.md) |
| [OpusMT-Zh-En](https://aihub.qualcomm.com/models/opus_mt_zh_en) | [qai_hub_models.models.opus_mt_zh_en](qai_hub_models/models/opus_mt_zh_en/README.md) |
| [TrOCR](https://aihub.qualcomm.com/models/trocr) | [qai_hub_models.models.trocr](qai_hub_models/models/trocr/README.md) |

### Audio

| Model | README |
| -- | -- |
| | |
| **Speech Recognition**
| [HuggingFace-WavLM-Base-Plus](https://aihub.qualcomm.com/models/huggingface_wavlm_base_plus) | [qai_hub_models.models.huggingface_wavlm_base_plus](qai_hub_models/models/huggingface_wavlm_base_plus/README.md) |
| [Whisper-Base](https://aihub.qualcomm.com/models/whisper_base) | [qai_hub_models.models.whisper_base](qai_hub_models/models/whisper_base/README.md) |
| [Whisper-Large-V3-Turbo](https://aihub.qualcomm.com/models/whisper_large_v3_turbo) | [qai_hub_models.models.whisper_large_v3_turbo](qai_hub_models/models/whisper_large_v3_turbo/README.md) |
| [Whisper-Small](https://aihub.qualcomm.com/models/whisper_small) | [qai_hub_models.models.whisper_small](qai_hub_models/models/whisper_small/README.md) |
| [Whisper-Small-Quantized](https://aihub.qualcomm.com/models/whisper_small_quantized) | [qai_hub_models.models.whisper_small_quantized](qai_hub_models/models/whisper_small_quantized/README.md) |
| [Whisper-Tiny](https://aihub.qualcomm.com/models/whisper_tiny) | [qai_hub_models.models.whisper_tiny](qai_hub_models/models/whisper_tiny/README.md) |
| [Zipformer](https://aihub.qualcomm.com/models/zipformer) | [qai_hub_models.models.zipformer](qai_hub_models/models/zipformer/README.md) |
| | |
| **Audio Classification**
| [YamNet](https://aihub.qualcomm.com/models/yamnet) | [qai_hub_models.models.yamnet](qai_hub_models/models/yamnet/README.md) |
| | |
| **Audio Generation**
| [MeloTTS-EN](https://aihub.qualcomm.com/models/melotts_en) | [qai_hub_models.models.melotts_en](qai_hub_models/models/melotts_en/README.md) |

### Generative AI

| Model | README |
| -- | -- |
| | |
| **Image Generation**
| [ControlNet-Canny](https://aihub.qualcomm.com/models/controlnet_canny) | [qai_hub_models.models.controlnet_canny](qai_hub_models/models/controlnet_canny/README.md) |
| [Stable-Diffusion-v1.5](https://aihub.qualcomm.com/models/stable_diffusion_v1_5) | [qai_hub_models.models.stable_diffusion_v1_5](qai_hub_models/models/stable_diffusion_v1_5/README.md) |
| [Stable-Diffusion-v2.1](https://aihub.qualcomm.com/models/stable_diffusion_v2_1) | [qai_hub_models.models.stable_diffusion_v2_1](qai_hub_models/models/stable_diffusion_v2_1/README.md) |
| | |
| **Text Generation**
| [Albert-Base-V2-Hf](https://aihub.qualcomm.com/models/albert_base_v2_hf) | [qai_hub_models.models.albert_base_v2_hf](qai_hub_models/models/albert_base_v2_hf/README.md) |
| [Baichuan2-7B](https://aihub.qualcomm.com/models/baichuan2_7b) | [qai_hub_models.models.baichuan2_7b](qai_hub_models/models/baichuan2_7b/README.md) |
| [Bert-Base-Uncased-Hf](https://aihub.qualcomm.com/models/bert_base_uncased_hf) | [qai_hub_models.models.bert_base_uncased_hf](qai_hub_models/models/bert_base_uncased_hf/README.md) |
| [Distil-Bert-Base-Uncased-Hf](https://aihub.qualcomm.com/models/distil_bert_base_uncased_hf) | [qai_hub_models.models.distil_bert_base_uncased_hf](qai_hub_models/models/distil_bert_base_uncased_hf/README.md) |
| [Electra-Bert-Base-Discrim-Google](https://aihub.qualcomm.com/models/electra_bert_base_discrim_google) | [qai_hub_models.models.electra_bert_base_discrim_google](qai_hub_models/models/electra_bert_base_discrim_google/README.md) |
| [Falcon3-7B-Instruct](https://aihub.qualcomm.com/models/falcon_v3_7b_instruct) | [qai_hub_models.models.falcon_v3_7b_instruct](qai_hub_models/models/falcon_v3_7b_instruct/README.md) |
| [IBM-Granite-v3.1-8B-Instruct](https://aihub.qualcomm.com/models/ibm_granite_v3_1_8b_instruct) | [qai_hub_models.models.ibm_granite_v3_1_8b_instruct](qai_hub_models/models/ibm_granite_v3_1_8b_instruct/README.md) |
| [IndusQ-1.1B](https://aihub.qualcomm.com/models/indus_1b) | [qai_hub_models.models.indus_1b](qai_hub_models/models/indus_1b/README.md) |
| [JAIS-6p7b-Chat](https://aihub.qualcomm.com/models/jais_6p7b_chat) | [qai_hub_models.models.jais_6p7b_chat](qai_hub_models/models/jais_6p7b_chat/README.md) |
| [Llama-SEA-LION-v3.5-8B-R](https://aihub.qualcomm.com/models/llama_v3_1_sea_lion_3_5_8b_r) | [qai_hub_models.models.llama_v3_1_sea_lion_3_5_8b_r](qai_hub_models/models/llama_v3_1_sea_lion_3_5_8b_r/README.md) |
| [Llama-v2-7B-Chat](https://aihub.qualcomm.com/models/llama_v2_7b_chat) | [qai_hub_models.models.llama_v2_7b_chat](qai_hub_models/models/llama_v2_7b_chat/README.md) |
| [Llama-v3-8B-Instruct](https://aihub.qualcomm.com/models/llama_v3_8b_instruct) | [qai_hub_models.models.llama_v3_8b_instruct](qai_hub_models/models/llama_v3_8b_instruct/README.md) |
| [Llama-v3-ELYZA-JP-8B](https://aihub.qualcomm.com/models/llama_v3_elyza_jp_8b) | [qai_hub_models.models.llama_v3_elyza_jp_8b](qai_hub_models/models/llama_v3_elyza_jp_8b/README.md) |
| [Llama-v3.1-8B-Instruct](https://aihub.qualcomm.com/models/llama_v3_1_8b_instruct) | [qai_hub_models.models.llama_v3_1_8b_instruct](qai_hub_models/models/llama_v3_1_8b_instruct/README.md) |
| [Llama-v3.2-1B-Instruct](https://aihub.qualcomm.com/models/llama_v3_2_1b_instruct) | [qai_hub_models.models.llama_v3_2_1b_instruct](qai_hub_models/models/llama_v3_2_1b_instruct/README.md) |
| [Llama-v3.2-3B-Instruct](https://aihub.qualcomm.com/models/llama_v3_2_3b_instruct) | [qai_hub_models.models.llama_v3_2_3b_instruct](qai_hub_models/models/llama_v3_2_3b_instruct/README.md) |
| [Llama3-TAIDE-LX-8B-Chat-Alpha1](https://aihub.qualcomm.com/models/llama_v3_taide_8b_chat) | [qai_hub_models.models.llama_v3_taide_8b_chat](qai_hub_models/models/llama_v3_taide_8b_chat/README.md) |
| [Mistral-7B-Instruct-v0.3](https://aihub.qualcomm.com/models/mistral_7b_instruct_v0_3) | [qai_hub_models.models.mistral_7b_instruct_v0_3](qai_hub_models/models/mistral_7b_instruct_v0_3/README.md) |
| [Mobile-Bert-Uncased-Google](https://aihub.qualcomm.com/models/mobile_bert_uncased_google) | [qai_hub_models.models.mobile_bert_uncased_google](qai_hub_models/models/mobile_bert_uncased_google/README.md) |
| [PLaMo-1B](https://aihub.qualcomm.com/models/plamo_1b) | [qai_hub_models.models.plamo_1b](qai_hub_models/models/plamo_1b/README.md) |
| [Phi-3.5-Mini-Instruct](https://aihub.qualcomm.com/models/phi_3_5_mini_instruct) | [qai_hub_models.models.phi_3_5_mini_instruct](qai_hub_models/models/phi_3_5_mini_instruct/README.md) |
| [Qwen2-7B-Instruct](https://aihub.qualcomm.com/models/qwen2_7b_instruct) | [qai_hub_models.models.qwen2_7b_instruct](qai_hub_models/models/qwen2_7b_instruct/README.md) |
| [Qwen2.5-7B-Instruct](https://aihub.qualcomm.com/models/qwen2_5_7b_instruct) | [qai_hub_models.models.qwen2_5_7b_instruct](qai_hub_models/models/qwen2_5_7b_instruct/README.md) |


## Need help?
Slack: https://aihub.qualcomm.com/community/slack

GitHub Issues: https://github.com/quic/ai-hub-models/issues

Email: ai-hub-support@qti.qualcomm.com.

## LICENSE

Qualcomm® AI Hub Models is licensed under BSD-3. See the [LICENSE file](LICENSE).
