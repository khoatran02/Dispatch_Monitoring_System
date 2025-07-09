# Object Detection and Classification API

A production-ready system for detecting and classifying dishes and trays in images/videos using YOLO and custom classifiers, with FastAPI backend.

## How It Works

The system processes visual input in three stages:

1. **Region of Interest Filtering**:
   - Focuses only on a predefined polygonal observation region
   - Uses masking to ignore irrelevant parts of the image

2. **Object Detection**:
   - YOLOv8 model detects dishes and trays within the observation region
   - Returns bounding boxes and object types

3. **Classification**:
   - MobileNetV3 classifiers determine dish/tray states:
     - Dish: `empty`, `kakigori`, `not_empty`
     - Tray: (custom states as defined in your model)
   - Returns classification confidence scores

![deepseek_mermaid_20250709_1d87ce](https://github.com/user-attachments/assets/7189df40-6f30-4a44-8fe0-aa2cf05596ec)


## Prerequisites

### Hardware
- NVIDIA GPU
- CUDA 11.7+ compatible drivers

### Software
1. **Docker** ([Install Guide](https://docs.docker.com/engine/install/))
2. **NVIDIA Container Toolkit** ([Install Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))
3. **docker-compose** v1.28+

Verify installations:
```bash
docker --version
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```



