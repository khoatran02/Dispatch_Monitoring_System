import cv2
import numpy as np
from datetime import datetime
import torch
from torchvision import models
import torch.nn as nn
from ultralytics import YOLO
from model import *


def process_image_with_detections(input_image, detector, classifier_dish, classifier_tray, observation_region):
    """
    Process ONLY the observation region of an image for detection and classification
    
    Args:
        input_image: Input image (numpy array or file path)
        detector: YOLO detection model
        classifier_dish: Dish classification model
        classifier_tray: Tray classification model
        observation_region: List of polygons defining the region of interest
        
    Returns:
        tuple: (labeled_output_image, list_of_classified_objects)
    """
    # Load image if path is provided
    if isinstance(input_image, str):
        frame = cv2.imread(input_image)
        if frame is None:
            raise ValueError(f"Could not read image {input_image}")
    else:
        frame = input_image.copy()
    
    # Create mask for observation region
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, observation_region, 255) # type: ignore
    
    # Crop to observation region
    x,y,w,h = cv2.boundingRect(observation_region[0])
    cropped_region = frame[y:y+h, x:x+w]
    region_mask = mask[y:y+h, x:x+w]
    
    # Apply mask to get only the polygonal region
    observation_area = cv2.bitwise_and(cropped_region, cropped_region, mask=region_mask)
    
    # Create output frame with observation region highlighted
    output_frame = frame.copy()
    overlay = output_frame.copy()
    cv2.fillPoly(overlay, observation_region, (0, 255, 255, 100))
    cv2.addWeighted(overlay, 0.3, output_frame, 0.7, 0, output_frame)
    
    # Detection ONLY in observation area
    detections = detector(observation_area, imgsz=320, half=True)[0]
    boxes = detections.boxes.xyxy.cpu().numpy()
    classes = detections.boxes.cls.cpu().numpy()
    confs = detections.boxes.conf.cpu().numpy()
    
    classified_objects = []
    
    # Process detections (all are within observation region by design)
    for box, cls, conf in zip(boxes, classes, confs):
        # Convert box coordinates back to original image space
        orig_x1 = int(box[0]) + x
        orig_y1 = int(box[1]) + y
        orig_x2 = int(box[2]) + x
        orig_y2 = int(box[3]) + y
        
        roi = frame[orig_y1:orig_y2, orig_x1:orig_x2]
        
        try:
            # Classification
            if cls == 0:  # Dish
                label = classifier_dish.predict(roi)
            else:  # Tray
                label = classifier_tray.predict(roi)
            
            # Store results
            classified_objects.append({
                "type": detector.names[int(cls)],
                "label": label,
                "confidence": float(conf),
                "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                "roi": roi
            })
            
            # Draw on original output frame
            color = (0, 255, 0)  # Green
            cv2.rectangle(output_frame, (orig_x1, orig_y1), (orig_x2, orig_y2), color, 2)
            cv2.putText(output_frame, 
                       f"{detector.names[int(cls)]}: {label} ({conf:.2f})", 
                       (orig_x1, orig_y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        except Exception as e:
            print(f"Error processing object: {e}")
            continue
    
    return output_frame, classified_objects


def process_video(input_video_path, output_video_path, detector, classifier_dish, classifier_tray, observation_region):

    # Initialize video capture
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video {input_video_path}")

    # Get video properties for output
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Processing loop
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        output_image, objects = process_image_with_detections(
            frame, 
            detector, 
            classifier_dish, 
            classifier_tray, 
            observation_region
        )
        
        
        # Write frame to output video
        out.write(frame)

    # Cleanup
    cap.release()
    out.release()
    print(f"Processing complete. Saved to {output_video_path}")


# Example usage
if __name__ == "__main__":
    # Initialize models
    detector = YOLO("model/detect.pt").to("cuda")
    classifier_dish = Classifier(model_name='MobileNetV3').load_model("model/dish.pth")
    classifier_tray = Classifier(model_name='MobileNetV3').load_model("model/tray.pth")
    
    # Define observation region
    observation_region = [
        np.array([[956, 64], [1156, 114], [1361, 176], [1481, 222], 
                 [1500, 274], [1483, 293], [1230, 229], [932, 150]])
    ]

    # Process video
    # input_video_path = "1473_CH05_20250501133703_154216.mp4"
    # output_video_path = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    # process_video(input_video_path, output_video_path, detector, classifier_dish, classifier_tray, observation_region)


    # Process single image
    input_image = "frame.png"  # or numpy array
    output_image, objects = process_image_with_detections(
        input_image, 
        detector, 
        classifier_dish, 
        classifier_tray, 
        observation_region
    )

    # Save results
    cv2.imwrite("output.jpg", output_image)

    # Print detected objects
    print("\nDetected Objects in Region:")
    for i, obj in enumerate(objects, 1):
        print(f"{i}. Type: {obj['type']}, Label: {obj['label']}, "
                f"Confidence: {obj['confidence']:.2f}, BBox: {obj['bbox']}")
        
        # Optionally save each cropped ROI
        cv2.imwrite(f"object_{i}_{obj['type']}_{obj['label']}.jpg", obj["roi"])