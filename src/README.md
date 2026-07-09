# [Qualcomm® AI Hub Models](https://aihub.qualcomm.com/)

[![Release](https://img.shields.io/github/v/release/quic/ai-hub-models)](https://github.com/qualcomm/ai-hub-models/releases/latest)
[![Tag](https://img.shields.io/github/v/tag/quic/ai-hub-models)](https://github.com/qualcomm/ai-hub-models/releases/latest)
[![PyPi](https://img.shields.io/pypi/v/qai-hub-models)](https://pypi.org/project/qai-hub-models/)
![Python 3.10, 3.11, 3.12, 3.13](https://img.shields.io/badge/python-3.10%20(Recommended)%2C%203.11%2C%203.12%2C%203.13-yellow)

The Qualcomm® AI Hub Models are a collection of
state-of-the-art machine learning models optimized for deployment on Qualcomm® devices.

* [List of Models by Category](#model-directory)
* [On-Device Performance Data](https://aihub.qualcomm.com/models)
* [Device-Native Sample Apps](https://github.com/qualcomm/ai-hub-apps)

See supported: [On-Device Runtimes](#on-device-runtimes), [Hardware Targets & Precision](#device-hardware--precision), [Chipsets](#chipsets), [Devices](#devices)

&nbsp;

## NEW: Quick Start with the AI Hub Models CLI

Use our lightweight command-line interface to browse and download from the collection of Qualcomm® AI Hub Models.


```shell
pip install qai_hub_models_cli # (the CLI is also available with the qai-hub-models package)

qai-hub-models models                                                # browse the catalog
qai-hub-models info mobilenet_v2                                     # model details + download options
qai-hub-models fetch mobilenet_v2 --runtime tflite --precision float # download a deployable asset
# ... and more
```

The CLI also offers a Python API.

See the [CLI README](cli/README.md) for full usage instructions.

&nbsp;

## Setup

### 1. Install Python Package

The package is available via pip:

```shell
# NOTE for Snapdragon X Elite and Snapdragon X2 Elite users:
# Only AMDx64 (64-bit) Python is supported on Windows.
# Installation will fail when using Windows ARM64 Python.

pip install qai_hub_models
```

Some models (e.g. YOLOv7) require
additional dependencies. See the model's install instructions in the [GitHub repository](https://github.com/qualcomm/ai-hub-models).

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

qai-hub-models export yolov7 --target-runtime tflite --precision float --device "Samsung Galaxy S25 (Family)"
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

**Native** applications that can run our models (with pre- and post-processing) on physical devices are published in the [AI Hub Apps repository](https://github.com/qualcomm/ai-hub-apps/).

**Python** applications are defined for all models (from qai_hub_models.models.\<model_name> import App). These apps wrap model inference with pre- and post-processing steps written using torch & numpy. **These apps are optimized to be an easy-to-follow example, rather than to minimize prediction time.**

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
* Snapdragon [8 Elite Gen 5](https://www.qualcomm.com/smartphones/products/8-series/snapdragon-8-elite-gen-5), [8 Elite](https://www.qualcomm.com/products/mobile/snapdragon/smartphones/snapdragon-8-series-mobile-platforms/snapdragon-8-elite-mobile-platform), [8 Gen 3](https://www.qualcomm.com/products/mobile/snapdragon/smartphones/snapdragon-8-series-mobile-platforms/snapdragon-8-gen-3-mobile-platform), [8 Gen 2](https://www.qualcomm.com/products/mobile/snapdragon/smartphones/snapdragon-8-series-mobile-platforms/snapdragon-8-gen-2-mobile-platform), and [8 Gen 1](https://www.qualcomm.com/products/mobile/snapdragon/smartphones/snapdragon-8-series-mobile-platforms/snapdragon-8-gen-1-mobile-platform) Mobile Platforms
* [Snapdragon X2 Elite](https://www.qualcomm.com/laptops/products/snapdragon-x2-elite), [Snapdragon X Elite](https://www.qualcomm.com/products/mobile/snapdragon/pcs-and-tablets/snapdragon-x-elite) Compute Platforms
* SA7255P, SA8295P, and SA8775P Automotive Platforms
* [QCS 6490](https://www.qualcomm.com/products/internet-of-things/industrial/building-enterprise/qcs6490), [QCS 8250](https://www.qualcomm.com/products/internet-of-things/consumer/cameras/qcs8250), [QCS 9075](https://www.qualcomm.com/internet-of-things/products/iq9-series/iq-9075), and [QCS 8550](https://www.qualcomm.com/products/technology/processors/qcs8550) IoT Platforms
* QCS8450 XR Platform

and many more.

### Devices
* Samsung Galaxy S21, S22, S23, S24, and S25 Series
* Xiaomi 12, 13, 15, and 17
* Snapdragon X Elite CRD and Snapdragon X2 Elite CRD (Compute Reference Device)
* Qualcomm RB3 Gen 2, RB5 Gen 2, IQ-8, IQ-9

and many more.

&nbsp;

## Model Directory

### Computer Vision

| Model | Package |
| -- | -- |
| | |
| **Image Classification**
| [Beit](https://aihub.qualcomm.com/models/beit) | `qai_hub_models.models.beit` |
| [ConvNext-Base](https://aihub.qualcomm.com/models/convnext_base) | `qai_hub_models.models.convnext_base` |
| [ConvNext-Tiny](https://aihub.qualcomm.com/models/convnext_tiny) | `qai_hub_models.models.convnext_tiny` |
| [DLA-102-X](https://aihub.qualcomm.com/models/dla102x) | `qai_hub_models.models.dla102x` |
| [DenseNet-121](https://aihub.qualcomm.com/models/densenet121) | `qai_hub_models.models.densenet121` |
| [EfficientFormer](https://aihub.qualcomm.com/models/efficientformer) | `qai_hub_models.models.efficientformer` |
| [EfficientNet-B0](https://aihub.qualcomm.com/models/efficientnet_b0) | `qai_hub_models.models.efficientnet_b0` |
| [EfficientNet-B4](https://aihub.qualcomm.com/models/efficientnet_b4) | `qai_hub_models.models.efficientnet_b4` |
| [EfficientNet-Lite4](https://aihub.qualcomm.com/models/efficientnet_lite4) | `qai_hub_models.models.efficientnet_lite4` |
| [EfficientNet-V2-s](https://aihub.qualcomm.com/models/efficientnet_v2_s) | `qai_hub_models.models.efficientnet_v2_s` |
| [EfficientViT-b2-cls](https://aihub.qualcomm.com/models/efficientvit_b2_cls) | `qai_hub_models.models.efficientvit_b2_cls` |
| [EfficientViT-l2-cls](https://aihub.qualcomm.com/models/efficientvit_l2_cls) | `qai_hub_models.models.efficientvit_l2_cls` |
| [GPUNet](https://aihub.qualcomm.com/models/gpunet) | `qai_hub_models.models.gpunet` |
| [GoogLeNet](https://aihub.qualcomm.com/models/googlenet) | `qai_hub_models.models.googlenet` |
| [Inception-v3](https://aihub.qualcomm.com/models/inception_v3) | `qai_hub_models.models.inception_v3` |
| [InternImage](https://aihub.qualcomm.com/models/internimage) | `qai_hub_models.models.internimage` |
| [LeViT](https://aihub.qualcomm.com/models/levit) | `qai_hub_models.models.levit` |
| [MNASNet05](https://aihub.qualcomm.com/models/mnasnet05) | `qai_hub_models.models.mnasnet05` |
| [Mobile-VIT](https://aihub.qualcomm.com/models/mobile_vit) | `qai_hub_models.models.mobile_vit` |
| [MobileNet-v2](https://aihub.qualcomm.com/models/mobilenet_v2) | `qai_hub_models.models.mobilenet_v2` |
| [MobileNet-v3-Large](https://aihub.qualcomm.com/models/mobilenet_v3_large) | `qai_hub_models.models.mobilenet_v3_large` |
| [MobileNet-v3-Small](https://aihub.qualcomm.com/models/mobilenet_v3_small) | `qai_hub_models.models.mobilenet_v3_small` |
| [NASNet](https://aihub.qualcomm.com/models/nasnet) | `qai_hub_models.models.nasnet` |
| [RegNet](https://aihub.qualcomm.com/models/regnet) | `qai_hub_models.models.regnet` |
| [RegNet-Y-800MF](https://aihub.qualcomm.com/models/regnet_y_800mf) | `qai_hub_models.models.regnet_y_800mf` |
| [ResNeXt101](https://aihub.qualcomm.com/models/resnext101) | `qai_hub_models.models.resnext101` |
| [ResNeXt50](https://aihub.qualcomm.com/models/resnext50) | `qai_hub_models.models.resnext50` |
| [ResNet101](https://aihub.qualcomm.com/models/resnet101) | `qai_hub_models.models.resnet101` |
| [ResNet18](https://aihub.qualcomm.com/models/resnet18) | `qai_hub_models.models.resnet18` |
| [ResNet50](https://aihub.qualcomm.com/models/resnet50) | `qai_hub_models.models.resnet50` |
| [Sequencer2D](https://aihub.qualcomm.com/models/sequencer2d) | `qai_hub_models.models.sequencer2d` |
| [Shufflenet-v2](https://aihub.qualcomm.com/models/shufflenet_v2) | `qai_hub_models.models.shufflenet_v2` |
| [SqueezeNet-1.1](https://aihub.qualcomm.com/models/squeezenet1_1) | `qai_hub_models.models.squeezenet1_1` |
| [Swin-Base](https://aihub.qualcomm.com/models/swin_base) | `qai_hub_models.models.swin_base` |
| [Swin-Small](https://aihub.qualcomm.com/models/swin_small) | `qai_hub_models.models.swin_small` |
| [Swin-Tiny](https://aihub.qualcomm.com/models/swin_tiny) | `qai_hub_models.models.swin_tiny` |
| [SwinV2-Base](https://aihub.qualcomm.com/models/swinv2_base) | `qai_hub_models.models.swinv2_base` |
| [VIT](https://aihub.qualcomm.com/models/vit) | `qai_hub_models.models.vit` |
| [WideResNet50](https://aihub.qualcomm.com/models/wideresnet50) | `qai_hub_models.models.wideresnet50` |
| | |
| **Image Editing**
| [AOT-GAN](https://aihub.qualcomm.com/models/aotgan) | `qai_hub_models.models.aotgan` |
| [DDColor](https://aihub.qualcomm.com/models/ddcolor) | `qai_hub_models.models.ddcolor` |
| [DnCNN](https://aihub.qualcomm.com/models/dncnn) | `qai_hub_models.models.dncnn` |
| [LaMa-Dilated](https://aihub.qualcomm.com/models/lama_dilated) | `qai_hub_models.models.lama_dilated` |
| [NAFNet-DeBlur](https://aihub.qualcomm.com/models/nafnet_deblur) | `qai_hub_models.models.nafnet_deblur` |
| [NAFNet-DeNoise](https://aihub.qualcomm.com/models/nafnet_denoise) | `qai_hub_models.models.nafnet_denoise` |
| | |
| **Super Resolution**
| [ESRGAN](https://aihub.qualcomm.com/models/esrgan) | `qai_hub_models.models.esrgan` |
| [NAFSSR](https://aihub.qualcomm.com/models/nafssr) | `qai_hub_models.models.nafssr` |
| [QuickSRNetLarge](https://aihub.qualcomm.com/models/quicksrnetlarge) | `qai_hub_models.models.quicksrnetlarge` |
| [QuickSRNetMedium](https://aihub.qualcomm.com/models/quicksrnetmedium) | `qai_hub_models.models.quicksrnetmedium` |
| [QuickSRNetSmall](https://aihub.qualcomm.com/models/quicksrnetsmall) | `qai_hub_models.models.quicksrnetsmall` |
| [Real-ESRGAN-General-x4v3](https://aihub.qualcomm.com/models/real_esrgan_general_x4v3) | `qai_hub_models.models.real_esrgan_general_x4v3` |
| [Real-ESRGAN-x4plus](https://aihub.qualcomm.com/models/real_esrgan_x4plus) | `qai_hub_models.models.real_esrgan_x4plus` |
| [SESR-M5](https://aihub.qualcomm.com/models/sesr_m5) | `qai_hub_models.models.sesr_m5` |
| [XLSR](https://aihub.qualcomm.com/models/xlsr) | `qai_hub_models.models.xlsr` |
| | |
| **Semantic Segmentation**
| [DDRNet23-Slim](https://aihub.qualcomm.com/models/ddrnet23_slim) | `qai_hub_models.models.ddrnet23_slim` |
| [DeepLabV3-Plus-MobileNet](https://aihub.qualcomm.com/models/deeplabv3_plus_mobilenet) | `qai_hub_models.models.deeplabv3_plus_mobilenet` |
| [DeepLabXception](https://aihub.qualcomm.com/models/deeplab_xception) | `qai_hub_models.models.deeplab_xception` |
| [EdgeTAM](https://aihub.qualcomm.com/models/edgetam) | `qai_hub_models.models.edgetam` |
| [FCN-ResNet50](https://aihub.qualcomm.com/models/fcn_resnet50) | `qai_hub_models.models.fcn_resnet50` |
| [FFNet-122NS-LowRes](https://aihub.qualcomm.com/models/ffnet_122ns_lowres) | `qai_hub_models.models.ffnet_122ns_lowres` |
| [FFNet-40S](https://aihub.qualcomm.com/models/ffnet_40s) | `qai_hub_models.models.ffnet_40s` |
| [FFNet-54S](https://aihub.qualcomm.com/models/ffnet_54s) | `qai_hub_models.models.ffnet_54s` |
| [FFNet-78S](https://aihub.qualcomm.com/models/ffnet_78s) | `qai_hub_models.models.ffnet_78s` |
| [FFNet-78S-LowRes](https://aihub.qualcomm.com/models/ffnet_78s_lowres) | `qai_hub_models.models.ffnet_78s_lowres` |
| [FastSam-S](https://aihub.qualcomm.com/models/fastsam_s) | `qai_hub_models.models.fastsam_s` |
| [FastSam-X](https://aihub.qualcomm.com/models/fastsam_x) | `qai_hub_models.models.fastsam_x` |
| [HRNet-W48-OCR](https://aihub.qualcomm.com/models/hrnet_w48_ocr) | `qai_hub_models.models.hrnet_w48_ocr` |
| [Mask2Former](https://aihub.qualcomm.com/models/mask2former) | `qai_hub_models.models.mask2former` |
| [MaskRCNN](https://aihub.qualcomm.com/models/maskrcnn) | `qai_hub_models.models.maskrcnn` |
| [MediaPipe-Selfie-Segmentation](https://aihub.qualcomm.com/models/mediapipe_selfie) | `qai_hub_models.models.mediapipe_selfie` |
| [MobileSam](https://aihub.qualcomm.com/models/mobilesam) | `qai_hub_models.models.mobilesam` |
| [PSPNet](https://aihub.qualcomm.com/models/pspnet) | `qai_hub_models.models.pspnet` |
| [PidNet](https://aihub.qualcomm.com/models/pidnet) | `qai_hub_models.models.pidnet` |
| [PointNet](https://aihub.qualcomm.com/models/pointnet) | `qai_hub_models.models.pointnet` |
| [SINet](https://aihub.qualcomm.com/models/sinet) | `qai_hub_models.models.sinet` |
| [SalsaNext](https://aihub.qualcomm.com/models/salsanext) | `qai_hub_models.models.salsanext` |
| [Segformer-Base](https://aihub.qualcomm.com/models/segformer_base) | `qai_hub_models.models.segformer_base` |
| [Segment-Anything-Model-2](https://aihub.qualcomm.com/models/sam2) | `qai_hub_models.models.sam2` |
| [Segment-Anything-Model-3](https://aihub.qualcomm.com/models/sam3) | `qai_hub_models.models.sam3` |
| [Unet-Segmentation](https://aihub.qualcomm.com/models/unet_segmentation) | `qai_hub_models.models.unet_segmentation` |
| [YOLO26-Segmentation](https://aihub.qualcomm.com/models/yolo26_seg) | `qai_hub_models.models.yolo26_seg` |
| [YOLOE-Segmentation](https://aihub.qualcomm.com/models/yoloe_seg) | `qai_hub_models.models.yoloe_seg` |
| [YOLOv11-Segmentation](https://aihub.qualcomm.com/models/yolov11_seg) | `qai_hub_models.models.yolov11_seg` |
| [YOLOv8-Segmentation](https://aihub.qualcomm.com/models/yolov8_seg) | `qai_hub_models.models.yolov8_seg` |
| | |
| **Video Classification**
| [ResNet-2Plus1D](https://aihub.qualcomm.com/models/resnet_2plus1d) | `qai_hub_models.models.resnet_2plus1d` |
| [ResNet-3D](https://aihub.qualcomm.com/models/resnet_3d) | `qai_hub_models.models.resnet_3d` |
| [ResNet-Mixed-Convolution](https://aihub.qualcomm.com/models/resnet_mixed) | `qai_hub_models.models.resnet_mixed` |
| [Video-MAE](https://aihub.qualcomm.com/models/video_mae) | `qai_hub_models.models.video_mae` |
| | |
| **Video Generation**
| [First-Order-Motion-Model](https://aihub.qualcomm.com/models/fomm) | `qai_hub_models.models.fomm` |
| | |
| **Video Object Tracking**
| [Track-Anything](https://aihub.qualcomm.com/models/track_anything) | `qai_hub_models.models.track_anything` |
| | |
| **Object Detection**
| [3D-Deep-BOX](https://aihub.qualcomm.com/models/deepbox) | `qai_hub_models.models.deepbox` |
| [CavaFace](https://aihub.qualcomm.com/models/cavaface) | `qai_hub_models.models.cavaface` |
| [CenterNet-2D](https://aihub.qualcomm.com/models/centernet_2d) | `qai_hub_models.models.centernet_2d` |
| [Conditional-DETR-ResNet50](https://aihub.qualcomm.com/models/conditional_detr_resnet50) | `qai_hub_models.models.conditional_detr_resnet50` |
| [DETR-ResNet101](https://aihub.qualcomm.com/models/detr_resnet101) | `qai_hub_models.models.detr_resnet101` |
| [DETR-ResNet101-DC5](https://aihub.qualcomm.com/models/detr_resnet101_dc5) | `qai_hub_models.models.detr_resnet101_dc5` |
| [DETR-ResNet50](https://aihub.qualcomm.com/models/detr_resnet50) | `qai_hub_models.models.detr_resnet50` |
| [DETR-ResNet50-DC5](https://aihub.qualcomm.com/models/detr_resnet50_dc5) | `qai_hub_models.models.detr_resnet50_dc5` |
| [Detectron2-Detection](https://aihub.qualcomm.com/models/detectron2_detection) | `qai_hub_models.models.detectron2_detection` |
| [Facial-Attribute-Detection](https://aihub.qualcomm.com/models/face_attrib_net) | `qai_hub_models.models.face_attrib_net` |
| [HRNetFace](https://aihub.qualcomm.com/models/hrnet_face) | `qai_hub_models.models.hrnet_face` |
| [Lightweight-Face-Detection](https://aihub.qualcomm.com/models/face_det_lite) | `qai_hub_models.models.face_det_lite` |
| [MediaPipe-Face-Detection](https://aihub.qualcomm.com/models/mediapipe_face) | `qai_hub_models.models.mediapipe_face` |
| [MediaPipe-Hand-Detection](https://aihub.qualcomm.com/models/mediapipe_hand) | `qai_hub_models.models.mediapipe_hand` |
| [MediaPipe-Hand-Gesture-Recognition](https://aihub.qualcomm.com/models/mediapipe_hand_gesture) | `qai_hub_models.models.mediapipe_hand_gesture` |
| [PPE-Detection](https://aihub.qualcomm.com/models/gear_guard_net) | `qai_hub_models.models.gear_guard_net` |
| [Person-Foot-Detection](https://aihub.qualcomm.com/models/foot_track_net) | `qai_hub_models.models.foot_track_net` |
| [RF-DETR](https://aihub.qualcomm.com/models/rf_detr) | `qai_hub_models.models.rf_detr` |
| [RTMDet](https://aihub.qualcomm.com/models/rtmdet) | `qai_hub_models.models.rtmdet` |
| [ResNet34-SSD](https://aihub.qualcomm.com/models/resnet34_ssd1200) | `qai_hub_models.models.resnet34_ssd1200` |
| [YOLO26-Detection](https://aihub.qualcomm.com/models/yolo26_det) | `qai_hub_models.models.yolo26_det` |
| [YOLOv10-Detection](https://aihub.qualcomm.com/models/yolov10_det) | `qai_hub_models.models.yolov10_det` |
| [YOLOv11-Detection](https://aihub.qualcomm.com/models/yolov11_det) | `qai_hub_models.models.yolov11_det` |
| [YOLOv8-Detection](https://aihub.qualcomm.com/models/yolov8_det) | `qai_hub_models.models.yolov8_det` |
| [YOLOv8-OBB](https://aihub.qualcomm.com/models/yolov8_obb) | `qai_hub_models.models.yolov8_obb` |
| [YOLOv9-Detection](https://aihub.qualcomm.com/models/yolov9_det) | `qai_hub_models.models.yolov9_det` |
| [Yolo-R](https://aihub.qualcomm.com/models/yolor) | `qai_hub_models.models.yolor` |
| [Yolo-X](https://aihub.qualcomm.com/models/yolox) | `qai_hub_models.models.yolox` |
| [Yolo-v3](https://aihub.qualcomm.com/models/yolov3) | `qai_hub_models.models.yolov3` |
| [Yolo-v5](https://aihub.qualcomm.com/models/yolov5) | `qai_hub_models.models.yolov5` |
| [Yolo-v6](https://aihub.qualcomm.com/models/yolov6) | `qai_hub_models.models.yolov6` |
| [Yolo-v7](https://aihub.qualcomm.com/models/yolov7) | `qai_hub_models.models.yolov7` |
| | |
| **Pose Estimation**
| [CenterNet-Pose](https://aihub.qualcomm.com/models/centernet_pose) | `qai_hub_models.models.centernet_pose` |
| [Facial-Landmark-Detection](https://aihub.qualcomm.com/models/facemap_3dmm) | `qai_hub_models.models.facemap_3dmm` |
| [HRNetPose](https://aihub.qualcomm.com/models/hrnet_pose) | `qai_hub_models.models.hrnet_pose` |
| [LiteHRNet](https://aihub.qualcomm.com/models/litehrnet) | `qai_hub_models.models.litehrnet` |
| [MediaPipe-Pose-Estimation](https://aihub.qualcomm.com/models/mediapipe_pose) | `qai_hub_models.models.mediapipe_pose` |
| [Posenet-Mobilenet](https://aihub.qualcomm.com/models/posenet_mobilenet) | `qai_hub_models.models.posenet_mobilenet` |
| [RTMPose-Body2d](https://aihub.qualcomm.com/models/rtmpose_body2d) | `qai_hub_models.models.rtmpose_body2d` |
| [SixDRepNet](https://aihub.qualcomm.com/models/sixd_repnet) | `qai_hub_models.models.sixd_repnet` |
| [YOLO26-Pose](https://aihub.qualcomm.com/models/yolo26_pose) | `qai_hub_models.models.yolo26_pose` |
| [YOLOv11-Pose](https://aihub.qualcomm.com/models/yolov11_pose) | `qai_hub_models.models.yolov11_pose` |
| | |
| **Gaze Estimation**
| [EyeGaze](https://aihub.qualcomm.com/models/eyegaze) | `qai_hub_models.models.eyegaze` |
| | |
| **Depth Estimation**
| [CREStereo](https://aihub.qualcomm.com/models/crestereo) | `qai_hub_models.models.crestereo` |
| [Depth-Anything](https://aihub.qualcomm.com/models/depth_anything) | `qai_hub_models.models.depth_anything` |
| [Depth-Anything-V2](https://aihub.qualcomm.com/models/depth_anything_v2) | `qai_hub_models.models.depth_anything_v2` |
| [Depth-Anything-V3](https://aihub.qualcomm.com/models/depth_anything_v3) | `qai_hub_models.models.depth_anything_v3` |
| [Midas-V2](https://aihub.qualcomm.com/models/midas) | `qai_hub_models.models.midas` |
| [StereoNet](https://aihub.qualcomm.com/models/stereonet) | `qai_hub_models.models.stereonet` |
| | |
| **Driver Assistance**
| [BEVDet](https://aihub.qualcomm.com/models/bevdet) | `qai_hub_models.models.bevdet` |
| [BEVFusion](https://aihub.qualcomm.com/models/bevfusion_det) | `qai_hub_models.models.bevfusion_det` |
| [CVT](https://aihub.qualcomm.com/models/cvt) | `qai_hub_models.models.cvt` |
| [CenterNet-3D](https://aihub.qualcomm.com/models/centernet_3d) | `qai_hub_models.models.centernet_3d` |
| [CenterPoint](https://aihub.qualcomm.com/models/centerpoint) | `qai_hub_models.models.centerpoint` |
| [GKT](https://aihub.qualcomm.com/models/gkt) | `qai_hub_models.models.gkt` |
| [RangeNet-Plus-Plus](https://aihub.qualcomm.com/models/rangenet_plus_plus) | `qai_hub_models.models.rangenet_plus_plus` |
| [StateTransformer](https://aihub.qualcomm.com/models/statetransformer) | `qai_hub_models.models.statetransformer` |
| | |
| **Robotics**
| [ACT](https://aihub.qualcomm.com/models/act) | `qai_hub_models.models.act` |

### Multimodal

| Model | Package |
| -- | -- |
| | |
| [EasyOCR](https://aihub.qualcomm.com/models/easyocr) | `qai_hub_models.models.easyocr` |
| [GR00TN1.5](https://aihub.qualcomm.com/models/grootn15) | `qai_hub_models.models.grootn15` |
| [MiniLM-v2](https://aihub.qualcomm.com/models/minilm_v2) | `qai_hub_models.models.minilm_v2` |
| [Nomic-Embed-Text](https://aihub.qualcomm.com/models/nomic_embed_text) | `qai_hub_models.models.nomic_embed_text` |
| [OpenAI-Clip](https://aihub.qualcomm.com/models/openai_clip) | `qai_hub_models.models.openai_clip` |
| [OpusMT-En-Es](https://aihub.qualcomm.com/models/opus_mt_en_es) | `qai_hub_models.models.opus_mt_en_es` |
| [OpusMT-En-Zh](https://aihub.qualcomm.com/models/opus_mt_en_zh) | `qai_hub_models.models.opus_mt_en_zh` |
| [OpusMT-Es-En](https://aihub.qualcomm.com/models/opus_mt_es_en) | `qai_hub_models.models.opus_mt_es_en` |
| [OpusMT-Zh-En](https://aihub.qualcomm.com/models/opus_mt_zh_en) | `qai_hub_models.models.opus_mt_zh_en` |
| [Pi0.5](https://aihub.qualcomm.com/models/pi05) | `qai_hub_models.models.pi05` |
| [TrOCR](https://aihub.qualcomm.com/models/trocr) | `qai_hub_models.models.trocr` |

### Audio

| Model | Package |
| -- | -- |
| | |
| **Speech Recognition**
| [Distil-Whisper](https://aihub.qualcomm.com/models/distil_whisper) | `qai_hub_models.models.distil_whisper` |
| [Whisper-Base](https://aihub.qualcomm.com/models/whisper_base) | `qai_hub_models.models.whisper_base` |
| [Whisper-Large-V3-Turbo](https://aihub.qualcomm.com/models/whisper_large_v3_turbo) | `qai_hub_models.models.whisper_large_v3_turbo` |
| [Whisper-Medium](https://aihub.qualcomm.com/models/whisper_medium) | `qai_hub_models.models.whisper_medium` |
| [Whisper-Small](https://aihub.qualcomm.com/models/whisper_small) | `qai_hub_models.models.whisper_small` |
| [Whisper-Small-Quantized](https://aihub.qualcomm.com/models/whisper_small_quantized) | `qai_hub_models.models.whisper_small_quantized` |
| [Whisper-Tiny](https://aihub.qualcomm.com/models/whisper_tiny) | `qai_hub_models.models.whisper_tiny` |
| [Zipformer](https://aihub.qualcomm.com/models/zipformer) | `qai_hub_models.models.zipformer` |
| | |
| **Audio Classification**
| [YamNet](https://aihub.qualcomm.com/models/yamnet) | `qai_hub_models.models.yamnet` |
| | |
| **Audio Generation**
| [MeloTTS-EN](https://aihub.qualcomm.com/models/melotts_en) | `qai_hub_models.models.melotts_en` |
| [MeloTTS-ES](https://aihub.qualcomm.com/models/melotts_es) | `qai_hub_models.models.melotts_es` |
| [MeloTTS-ZH](https://aihub.qualcomm.com/models/melotts_zh) | `qai_hub_models.models.melotts_zh` |
| [PiperTTS-DE](https://aihub.qualcomm.com/models/pipertts_de) | `qai_hub_models.models.pipertts_de` |
| [PiperTTS-EN](https://aihub.qualcomm.com/models/pipertts_en) | `qai_hub_models.models.pipertts_en` |
| [PiperTTS-IT](https://aihub.qualcomm.com/models/pipertts_it) | `qai_hub_models.models.pipertts_it` |

### Generative AI

| Model | Package |
| -- | -- |
| | |
| **Image Generation**
| [ControlNet-Canny](https://aihub.qualcomm.com/models/controlnet_canny) | `qai_hub_models.models.controlnet_canny` |
| [Stable-Diffusion-v1.5](https://aihub.qualcomm.com/models/stable_diffusion_v1_5) | `qai_hub_models.models.stable_diffusion_v1_5` |
| [Stable-Diffusion-v2.1](https://aihub.qualcomm.com/models/stable_diffusion_v2_1) | `qai_hub_models.models.stable_diffusion_v2_1` |
| | |
| **Text Generation**
| [Albert-Base-V2-Hf](https://aihub.qualcomm.com/models/albert_base_v2_hf) | `qai_hub_models.models.albert_base_v2_hf` |
| [Bert-Base-Uncased-Hf](https://aihub.qualcomm.com/models/bert_base_uncased_hf) | `qai_hub_models.models.bert_base_uncased_hf` |
| [Distil-Bert-Base-Uncased-Hf](https://aihub.qualcomm.com/models/distil_bert_base_uncased_hf) | `qai_hub_models.models.distil_bert_base_uncased_hf` |
| [Electra-Bert-Base-Discrim-Google](https://aihub.qualcomm.com/models/electra_bert_base_discrim_google) | `qai_hub_models.models.electra_bert_base_discrim_google` |
| [Falcon3-7B-Instruct](https://aihub.qualcomm.com/models/falcon_v3_7b_instruct) | `qai_hub_models.models.falcon_v3_7b_instruct` |
| [GPT-OSS-20B](https://aihub.qualcomm.com/models/gpt_oss_20b) | `qai_hub_models.models.gpt_oss_20b` |
| [Gemma-4-E2B-it](https://aihub.qualcomm.com/models/gemma_4_e2b_it) | `qai_hub_models.models.gemma_4_e2b_it` |
| [Gemma-4-E4B-it](https://aihub.qualcomm.com/models/gemma_4_e4b_it) | `qai_hub_models.models.gemma_4_e4b_it` |
| [Granite-4.0-Micro](https://aihub.qualcomm.com/models/granite_4_0_micro) | `qai_hub_models.models.granite_4_0_micro` |
| [IBM-Granite-v3.1-8B-Instruct](https://aihub.qualcomm.com/models/ibm_granite_v3_1_8b_instruct) | `qai_hub_models.models.ibm_granite_v3_1_8b_instruct` |
| [IndusQ-1.1B](https://aihub.qualcomm.com/models/indus_1b) | `qai_hub_models.models.indus_1b` |
| [JAIS-6p7b-Chat](https://aihub.qualcomm.com/models/jais_6p7b_chat) | `qai_hub_models.models.jais_6p7b_chat` |
| [Llama-SEA-LION-v3.5-8B-R](https://aihub.qualcomm.com/models/llama_v3_1_sea_lion_3_5_8b_r) | `qai_hub_models.models.llama_v3_1_sea_lion_3_5_8b_r` |
| [Llama-v3-8B-Instruct](https://aihub.qualcomm.com/models/llama_v3_8b_instruct) | `qai_hub_models.models.llama_v3_8b_instruct` |
| [Llama-v3-ELYZA-JP-8B](https://aihub.qualcomm.com/models/llama_v3_elyza_jp_8b) | `qai_hub_models.models.llama_v3_elyza_jp_8b` |
| [Llama-v3.1-8B-Instruct](https://aihub.qualcomm.com/models/llama_v3_1_8b_instruct) | `qai_hub_models.models.llama_v3_1_8b_instruct` |
| [Llama-v3.2-1B-Instruct](https://aihub.qualcomm.com/models/llama_v3_2_1b_instruct) | `qai_hub_models.models.llama_v3_2_1b_instruct` |
| [Llama-v3.2-3B-Instruct](https://aihub.qualcomm.com/models/llama_v3_2_3b_instruct) | `qai_hub_models.models.llama_v3_2_3b_instruct` |
| [Llama-v3.2-3B-Instruct-SSD](https://aihub.qualcomm.com/models/llama_v3_2_3b_instruct_ssd) | `qai_hub_models.models.llama_v3_2_3b_instruct_ssd` |
| [Llama3-TAIDE-LX-8B-Chat-Alpha1](https://aihub.qualcomm.com/models/llama_v3_taide_8b_chat) | `qai_hub_models.models.llama_v3_taide_8b_chat` |
| [Ministral-3-3B-Instruct-2512](https://aihub.qualcomm.com/models/ministral_3_3b_instruct_2512) | `qai_hub_models.models.ministral_3_3b_instruct_2512` |
| [Mistral-7B-Instruct-v0.3](https://aihub.qualcomm.com/models/mistral_7b_instruct_v0_3) | `qai_hub_models.models.mistral_7b_instruct_v0_3` |
| [Mobile-Bert-Uncased-Google](https://aihub.qualcomm.com/models/mobile_bert_uncased_google) | `qai_hub_models.models.mobile_bert_uncased_google` |
| [PLaMo-1B](https://aihub.qualcomm.com/models/plamo_1b) | `qai_hub_models.models.plamo_1b` |
| [Phi-3.5-Mini-Instruct](https://aihub.qualcomm.com/models/phi_3_5_mini_instruct) | `qai_hub_models.models.phi_3_5_mini_instruct` |
| [Phi-4-Mini-Instruct](https://aihub.qualcomm.com/models/phi_4_mini_instruct) | `qai_hub_models.models.phi_4_mini_instruct` |
| [Qwen2-7B-Instruct](https://aihub.qualcomm.com/models/qwen2_7b_instruct) | `qai_hub_models.models.qwen2_7b_instruct` |
| [Qwen2.5-VL-7B-Instruct](https://aihub.qualcomm.com/models/qwen2_5_vl_7b_instruct) | `qai_hub_models.models.qwen2_5_vl_7b_instruct` |
| [Qwen3-0.6B](https://aihub.qualcomm.com/models/qwen3_0_6b) | `qai_hub_models.models.qwen3_0_6b` |
| [Qwen3-1.7B](https://aihub.qualcomm.com/models/qwen3_1_7b) | `qai_hub_models.models.qwen3_1_7b` |
| [Qwen3-4B](https://aihub.qualcomm.com/models/qwen3_4b) | `qai_hub_models.models.qwen3_4b` |
| [Qwen3-4B-Instruct-2507](https://aihub.qualcomm.com/models/qwen3_4b_instruct_2507) | `qai_hub_models.models.qwen3_4b_instruct_2507` |
| [Qwen3-8B](https://aihub.qualcomm.com/models/qwen3_8b) | `qai_hub_models.models.qwen3_8b` |
| [Qwen3-VL-2B-Instruct](https://aihub.qualcomm.com/models/qwen3_vl_2b_instruct) | `qai_hub_models.models.qwen3_vl_2b_instruct` |
| [Qwen3-VL-4B-Instruct](https://aihub.qualcomm.com/models/qwen3_vl_4b_instruct) | `qai_hub_models.models.qwen3_vl_4b_instruct` |
| [Qwen3.5-0.8B](https://aihub.qualcomm.com/models/qwen3_5_0_8b) | `qai_hub_models.models.qwen3_5_0_8b` |
| [Qwen3.5-2B](https://aihub.qualcomm.com/models/qwen3_5_2b) | `qai_hub_models.models.qwen3_5_2b` |

## Need help?
Slack: https://aihub.qualcomm.com/community/slack

GitHub Issues: https://github.com/qualcomm/ai-hub-models/issues

Email: ai-hub-support@qti.qualcomm.com.

## LICENSE

Qualcomm® AI Hub Models is licensed under BSD-3. See the [LICENSE file](https://github.com/qualcomm/ai-hub-models/blob/main/LICENSE).
