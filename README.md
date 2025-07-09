# Object Detection and Classification System

A production-ready pipeline for detecting and classifying dishes and trays in images/videos using YOLO and custom classifiers

Table of Contents
System Overview

Prerequisites

Installation

Configuration

Usage

Image Processing

Video Processing

API Mode (Optional)

Folder Structure

Customization

Troubleshooting

Performance Optimization

License

System Overview
This system provides:

✔ YOLO-based object detection (detect.pt)
✔ Custom classifiers for dishes (dish.pth) and trays (tray.pth)
✔ Region-of-interest filtering to focus on specific areas
✔ GPU-accelerated processing (NVIDIA required)
✔ Dockerized deployment for easy setup

Supported Models:

Detection: YOLOv8

Classification: MobileNetV3, ResNet18/50, EfficientNet-B3

Prerequisites
Hardware
NVIDIA GPU (Recommended: RTX 3060 or better)

CUDA 11.7+ compatible drivers

Software
Docker (Install Guide)

NVIDIA Container Toolkit (Install Guide)

docker-compose v1.28+

Verify installations:

bash
docker --version
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
Installation
1. Clone Repository
bash
git clone https://github.com/your-repo/object-detection-system.git
cd object-detection-system
2. Prepare Model Files
Place these in ./model/:

detect.pt (YOLO detection model)

dish.pth (Dish classifier)

tray.pth (Tray classifier)

3. Build & Launch
bash
docker-compose up -d --build
First build may take 5-10 minutes to download base images.

Configuration
Key Parameters (Edit in pipeline_system.py)
Parameter	Description	Example Value
observation_region	Polygon coordinates for ROI	[[[956,64], [1156,114], ...]]
model_name	Classifier architecture	'MobileNetV3'
imgsz	YOLO inference size	320
half	FP16 inference	True
Environment Variables
Add to docker-compose.yml if needed:

yaml
environment:
  - OMP_NUM_THREADS=4  # Controls CPU parallelism
Usage
Image Processing
Place images in ./data/input_images/

Run processing:

bash
docker exec detection-service python pipeline_system.py \
  --input ./data/input_images/frame.jpg \
  --output ./output/results/
Outputs:

results.jpg (Annotated image)

object_{n}_type_label.jpg (Individual cropped detections)

Video Processing
bash
docker exec detection-service python pipeline_system.py \
  --video ./data/input_videos/meal_service.mp4 \
  --output ./output/processed_videos/
API Mode (Optional)
Uncomment in pipeline_system.py:

python
# FastAPI implementation would go here
# @app.post("/process")
Folder Structure
text
.
├── data/                   # Input assets
│   ├── input_images/       # .jpg, .png
│   └── input_videos/       # .mp4, .avi
│
├── model/                  # ML models
│   ├── detect.pt           # YOLO weights
│   ├── dish.pth            # Dish classifier
│   └── tray.pth            # Tray classifier
│
├── output/                 # Processed results
│   ├── images/             # Annotated images
│   ├── videos/             # Processed videos
│   └── crops/              # Individual detections
│
├── web/                    # Optional UI
├── Dockerfile              # Container definition
├── docker-compose.yml      # Orchestration
└── requirements.txt        # Python dependencies
Customization
Adding New Classes
Retrain classifiers with updated model.py:

python
self.class_names = ['empty', 'kakigori', 'not_empty', 'new_class']  # ← Add here
Update label handling in pipeline_system.py:

python
if cls == 0:  # Dish
    label = classifier_dish.predict(roi)
elif cls == 1:  # Tray
    label = classifier_tray.predict(roi)
else:  # New detection type
    label = "unknown"
Troubleshooting
Issue	Solution
CUDA errors	Run nvidia-smi to verify GPU detection
Model loading fails	Check file paths in pipeline_system.py
Low FPS	Reduce imgsz or disable half=True
Memory errors	Decrease batch size in DataLoader
Logs:

bash
docker logs detection-service
Performance Optimization
For Jetson Devices
Add to Dockerfile:

dockerfile
ENV CUDA_VISIBLE_DEVICES=0
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
CPU-Only Mode
Modify docker-compose.yml:

yaml
services:
  detection-service:
    environment:
      - CUDA_VISIBLE_DEVICES=-1  # Disables GPU
License
Apache 2.0 - See LICENSE for details.

For commercial use or support, contact [your email].

Note: This README assumes you're using the provided pipeline_system.py implementation. Adjust paths/commands if your actual script differs.

