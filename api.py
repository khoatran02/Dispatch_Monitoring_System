# api.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import cv2
import numpy as np
import io
from datetime import datetime
from typing import List, Dict, Any
import torch
from ultralytics import YOLO
from model import Classifier
from uility import process_image_with_detections, process_video
import logging
import aiofiles
import os

# Initialize FastAPI app
app = FastAPI(
    title="Object Detection and Classification API",
    description="API for detecting and classifying dishes and trays in images/videos",
    version="1.0.0"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for loaded models
MODELS_LOADED = False
detector = None
classifier_dish = None
classifier_tray = None
observation_region = None

def initialize_models():
    """Initialize models on startup"""
    global detector, classifier_dish, classifier_tray, observation_region, MODELS_LOADED
    
    if not MODELS_LOADED:
        try:
            # Initialize models with error handling
            logger.info("Loading detection model...")
            detector = YOLO("model/detect.pt").to("cuda" if torch.cuda.is_available() else "cpu")
            
            logger.info("Loading dish classification model...")
            classifier_dish = Classifier(model_name='MobileNetV3').load_model("model/dish.pth")
            
            logger.info("Loading tray classification model...")
            classifier_tray = Classifier(model_name='MobileNetV3').load_model("model/tray.pth")
            
            # Define observation region (should be configurable)
            observation_region = [
                np.array([[956, 64], [1156, 114], [1361, 176], [1481, 222], 
                         [1500, 274], [1483, 293], [1230, 229], [932, 150]])
            ]
            
            MODELS_LOADED = True
            logger.info("All models loaded successfully")
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize models when the application starts"""
    initialize_models()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": MODELS_LOADED}

@app.post("/detect/image")
async def detect_objects_in_image(
    file: UploadFile = File(...),
    return_image: bool = True,
    confidence_threshold: float = 0.5
):
    """
    Process an image and return detected objects
    
    Args:
        file: Image file (JPEG, PNG)
        return_image: Whether to return annotated image
        confidence_threshold: Minimum confidence score for detections
        
    Returns:
        JSON with detection results and optionally the annotated image
    """
    if not MODELS_LOADED:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Read and validate image
        contents = await file.read()
        np_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Process image
        output_image, objects = process_image_with_detections(
            image,
            detector,
            classifier_dish,
            classifier_tray,
            observation_region
        )
        
        # Filter by confidence threshold
        filtered_objects = [
            obj for obj in objects if obj['confidence'] >= confidence_threshold
        ]
        
        # Prepare response
        response = {
            "timestamp": datetime.utcnow().isoformat(),
            "detections": filtered_objects,
            "detection_count": len(filtered_objects)
        }
        
        if return_image:
            # Convert output image to bytes
            _, encoded_image = cv2.imencode('.jpg', output_image)
            image_bytes = encoded_image.tobytes()
            
            return StreamingResponse(
                io.BytesIO(image_bytes),
                media_type="image/jpeg",
                headers={
                    "detection-count": str(len(filtered_objects)),
                    "X-Detection-Results": "success"
                }
            )
        
        return JSONResponse(content=response)
    
    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/detect/video")
async def detect_objects_in_video(
    file: UploadFile = File(...),
    output_format: str = "json",
    confidence_threshold: float = 0.5
):
    """
    Process a video and return detection results
    
    Args:
        file: Video file (MP4, AVI)
        output_format: 'json' for metadata or 'video' for annotated video
        confidence_threshold: Minimum confidence score for detections
        
    Returns:
        JSON with detection results or annotated video
    """
    if not MODELS_LOADED:
        raise HTTPException(status_code=503, detail="Models not loaded")
    

     

    # Save uploaded video temporarily
    temp_video_path = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    try:
        async with aiofiles.open(temp_video_path, "wb") as buffer:
            await buffer.write(await file.read())
    except Exception as e:
        logger.error(f"Failed to save uploaded video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded video: {str(e)}")
    
    # Process video based on output format
    try:
        if output_format == "json":
            return process_video_to_json(
                temp_video_path,
                confidence_threshold
            )
        elif output_format == "video":
            return process_video_to_annotated(
                temp_video_path
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid output format")
    except Exception as e:
        logger.error(f"Video processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)

def process_video_to_json(
    video_path: str,
    confidence_threshold: float
) -> JSONResponse:
    """Process video and return JSON results"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not read video file")
    
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "frames": [],
        "total_detections": 0
    }
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        _, objects = process_image_with_detections(
            frame,
            detector,
            classifier_dish,
            classifier_tray,
            observation_region
        )
        
        # Filter by confidence threshold
        filtered_objects = [
            obj for obj in objects if obj['confidence'] >= confidence_threshold
        ]
        
        results["frames"].append({
            "frame_number": frame_count,
            "detections": filtered_objects,
            "detection_count": len(filtered_objects)
        })
        results["total_detections"] += len(filtered_objects)
        
        frame_count += 1
    
    cap.release()
    return JSONResponse(content=results)

def process_video_to_annotated(video_path: str) -> StreamingResponse:
    """Process video and return annotated video stream"""
    output_path = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    
    try:
        process_video(
            video_path,
            output_path,
            detector,
            classifier_dish,
            classifier_tray,
            observation_region
        )
        
        def iterfile():
            with open(output_path, mode="rb") as file:
                yield from file
        
        return StreamingResponse(
            iterfile(),
            media_type="video/mp4",
            headers={
                "Content-Disposition": f"attachment; filename={output_path}"
            }
        )
    finally:
        import os
        if os.path.exists(output_path):
            os.remove(output_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)