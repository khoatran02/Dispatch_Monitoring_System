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

## Installation
1. Clone the repository:
```bash
git clone https://github.com/khoatran02/Dispatch_Monitoring_System.git
cd Dispatch_Monitoring_System
```
2. Prepare model files:
```text
model/
├── detect.pt     # YOLO detection model
├── dish.pth      # Dish classifier
└── tray.pth      # Tray classifier
```

3. Build and launch the services:
```bash
docker-compose up -d --build
```

# Running the Application

The system will automatically:

* Initialize all ML models
* Start the FastAPI server on port 8000
* (Optional) Launch web interface on port 8080

Check service status:

```bash
docker-compose ps
```
View logs:
```bash
docker-compose logs -f api-service
```

# API Usage

## Health Check
```bash
curl http://localhost:8000/health
```
Image Processing

```bash
curl -X POST -F "file=@test.jpg" http://localhost:8000/detect/image
```

Parameters:

* return_image: boolean (default true) - return annotated image
* confidence_threshold: float (default 0.5) - minimum detection confidence

Response:
* Returns JPEG image with annotations OR
* JSON with detection metadata

Video Processing

```bash
curl -X POST -F "file=@test.mp4" \-F "output_format=json" \http://localhost:8000/detect/video
```
Parameters:

* output_format: "json" or "video"
* confidence_threshold: float (default 0.5)
  
Response:

* JSON with frame-by-frame detections OR
* Annotated MP4 video stream










