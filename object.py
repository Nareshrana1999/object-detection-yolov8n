#!/usr/bin/env python3
"""
Object Detection using YOLOv8 with Ultralytics

This script performs object detection on images using YOLOv8 model from Ultralytics.
"""

import argparse
import os
import cv2
import numpy as np
import sys
from ultralytics import YOLO
from pathlib import Path
from typing import List, Dict, Any, Tuple

def load_model(model_name: str = 'yolov8n.pt') -> YOLO:
    """Load YOLOv8 model."""
    print(f"Loading {model_name} model...")
    try:
        model = YOLO(model_name)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting to download the model...")
        model = YOLO(model_name, task='detect')
        return model

def detect_objects(
    model: YOLO,
    img_path: str,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.4,
    img_size: int = 640
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Detect objects in an image using YOLOv8."""
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image at {img_path}")
    
    # Perform inference
    results = model(img, conf=conf_threshold, iou=iou_threshold, imgsz=img_size)
    
    # Parse results
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            cls_name = model.names[cls]
            
            detections.append({
                'class_id': cls,
                'class_name': cls_name,
                'confidence': conf,
                'bbox': [x1, y1, x2 - x1, y2 - y1],  # [x, y, w, h]
                'box_points': [x1, y1, x2, y2]  # [x1, y1, x2, y2]
            })
    
    return img, detections

def draw_detections(
    img: np.ndarray,
    detections: List[Dict[str, Any]],
    color: Tuple[int, int, int] = (0, 255, 0),
    show_conf: bool = True
) -> np.ndarray:
    """Draw bounding boxes and labels on the image."""
    img_with_boxes = img.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det['box_points']
        
        # Draw rectangle
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label
        label = f"{det['class_name']} {det['confidence']:.2f}" if show_conf else det['class_name']
        
        # Get text size
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Draw filled rectangle for label
        cv2.rectangle(img_with_boxes, (x1, y1 - 20), (x1 + text_w, y1), color, -1)
        
        # Put text
        cv2.putText(
            img_with_boxes,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),  # Black text
            1,
            cv2.LINE_AA
        )
    
    return img_with_boxes

def run_webcam_detection(model, conf_threshold=0.5, iou_threshold=0.4):
    """Run object detection on webcam feed."""
    print("Starting webcam detection...")
    print("Press 'q' to quit.")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    try:
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Perform detection
            results = model(frame, conf=conf_threshold, iou=iou_threshold)
            
            # Draw detections
            annotated_frame = results[0].plot()
            
            # Display the frame with detections
            cv2.imshow('YOLOv8 Webcam Detection', annotated_frame)
            
            # Break the loop on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

def get_user_choice():
    """Display interactive menu and get user choice."""
    print("\n" + "="*50)
    print("YOLOv8 Object Detection".center(50))
    print("="*50)
    print("\nSelect detection mode:")
    print("1. Image Detection (process an image file)")
    print("2. Webcam Detection (use your camera)")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (0-2): ")
            if choice in ['0', '1', '2']:
                return int(choice)
            print("Invalid choice. Please enter 0, 1, or 2.")
        except ValueError:
            print("Please enter a valid number.")

def get_image_path():
    """Prompt user for image path with validation."""
    while True:
        image_path = input("\nEnter path to image file: ").strip('"')
        if os.path.isfile(image_path):
            return image_path
        print(f"Error: File '{image_path}' not found. Please try again.")

def main():
    print("Starting YOLOv8 Object Detection...")
    parser = argparse.ArgumentParser(description='Object Detection using YOLOv8')
    parser.add_argument('--source', type=str, default=None,
                        help='Source for detection: image or webcam')
    parser.add_argument('--image', '-i', type=str, help='Path to input image')
    parser.add_argument('--output', '-o', type=str, default='output.jpg',
                        help='Path to save output image/video (default: output.jpg)')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold (default: 0.5)')
    parser.add_argument('--iou', type=float, default=0.4,
                        help='IOU threshold for NMS (default: 0.4)')
    parser.add_argument('--model', default='yolov8n.pt',
                        help='Model to use (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)')
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # If no source specified via command line, show interactive menu
    if args.source is None:
        choice = get_user_choice()
        
        if choice == 0:
            print("Exiting...")
            return
        elif choice == 1:  # Image mode
            args.source = 'image'
            args.image = get_image_path()
        else:  # Webcam mode
            args.source = 'webcam'
    
    # Print debug information
    print("\n" + "-"*50)
    print("Detection Settings:")
    print("-"*50)
    print(f"Mode: {'Image' if args.source == 'image' else 'Webcam'}")
    if args.source == 'image':
        print(f"Input image: {os.path.abspath(args.image) if args.image else 'Not provided'}")
    print(f"Output: {os.path.abspath(args.output)}")
    print(f"Model: {args.model}")
    print(f"Confidence threshold: {args.conf}")
    print(f"IOU threshold: {args.iou}")
    print("-"*50 + "\n")
    
    try:
        # Load model
        model = load_model(args.model)
        
        if args.source == 'webcam':
            # Run webcam detection
            run_webcam_detection(model, args.conf, args.iou)
        else:
            # Check if input file exists for image mode
            if not args.image:
                print("Error: --image argument is required when source is 'image'")
                return 1
                
            if not os.path.isfile(args.image):
                print(f"Error: Input file '{args.image}' not found!")
                return 1
            
            # Perform detection on image
            print(f"Processing image: {args.image}")
            img, detections = detect_objects(model, args.image, args.conf, args.iou)
            
            print(f"Found {len(detections)} objects")
            
            # Draw detections
            img_with_boxes = draw_detections(img, detections)
            
            # Save output
            cv2.imwrite(args.output, img_with_boxes)
            print(f"Output saved to {args.output}")
            
            # Show result
            cv2.imshow('Object Detection', img_with_boxes)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
