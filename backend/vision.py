"""
Vision processing using YOLOv8 for object detection and messiness scoring.
"""
import os
import cv2
import numpy as np
from PIL import Image
import base64
import io
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Global model instance
_model = None

# Object classes that indicate messiness
MESSY_OBJECTS = {
    # Household items
    'bottle', 'cup', 'bowl', 'fork', 'knife', 'spoon', 'plate',
    # Clothing
    'person', 'tie', 'backpack', 'handbag', 'suitcase',
    # Paper/trash
    'book', 'newspaper', 'magazine', 'paper',
    # Misc
    'potted plant', 'vase', 'remote', 'cell phone', 'keyboard',
    'mouse', 'laptop', 'tv', 'tie', 'umbrella'
}

# Object classes that are neutral (not necessarily messy)
NEUTRAL_OBJECTS = {
    'chair', 'couch', 'bed', 'dining table', 'toilet', 'sink',
    'bathtub', 'refrigerator', 'oven', 'microwave', 'washer',
    'lamp', 'light', 'window', 'door', 'floor'
}

# Minimum objects to consider messy
MESSY_THRESHOLD = 5

# Edge detection parameters for texture analysis
EDGE_THRESHOLD = 10000  # Minimum edges to consider messy


def get_model():
    """
    Get or initialize the YOLO model.
    """
    global _model
    
    if _model is not None:
        return _model
    
    try:
        from ultralytics import YOLO
        logger.info("Loading YOLOv8n model...")
        
        # Use YOLOv8n - smallest and fastest model
        # Will download automatically on first run
        _model = YOLO('yolov8n.pt')
        
        logger.info("YOLO model loaded successfully")
        return _model
        
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        raise


def detect_objects(frame_data: bytes) -> Tuple[List[Dict], Image.Image]:
    """
    Detect objects in the frame.
    
    Args:
        frame_data: Image bytes
        
    Returns:
        Tuple of (detected_objects_list, annotated_image)
    """
    try:
        # Decode image
        nparr = np.frombuffer(frame_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error("Failed to decode image")
            return [], None
        
        # Convert BGR to RGB for YOLO
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Run detection
        model = get_model()
        results = model(image, conf=0.3, verbose=False)
        
        # Parse results
        detections = []
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    box = boxes[i]
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = result.names[class_id]
                    
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': box.xyxy[0].cpu().numpy().tolist()
                    })
        
        # Draw annotations on image
        annotated = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.putText(annotated, label, (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert annotated to PIL
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        annotated_pil = Image.fromarray(annotated_rgb)
        
        return detections, annotated_pil
        
    except Exception as e:
        logger.error(f"Object detection error: {e}")
        return [], None


def analyze_messiness(detections: List[Dict], image: Image.Image = None) -> Dict:
    """
    Analyze messiness based on detected objects and image properties.
    
    Args:
        detections: List of detected objects
        image: PIL Image for texture analysis
        
    Returns:
        Dictionary with messiness score and level
    """
    try:
        # Count messy objects
        messy_count = 0
        neutral_count = 0
        
        for det in detections:
            class_name = det['class'].lower()
            
            # Check if it's a messy object
            if any(messy in class_name for messy in MESSY_OBJECTS):
                messy_count += 1
            elif any(neutral in class_name for neutral in NEUTRAL_OBJECTS):
                neutral_count += 1
        
        # Edge/texture analysis for messiness
        edge_score = 0
        if image:
            try:
                # Convert to grayscale
                img_array = np.array(image.convert('L'))
                
                # Detect edges
                edges = cv2.Canny(img_array, 50, 150)
                edge_count = np.sum(edges > 0)
                
                # Normalize edge score (higher = messier)
                total_pixels = img_array.size
                edge_ratio = edge_count / total_pixels
                edge_score = int(edge_ratio * 100)
                
            except Exception as e:
                logger.warning(f"Edge analysis failed: {e}")
        
        # Calculate messiness score (0-100)
        # Weight: object count + edge density
        object_score = min(50, messy_count * 10)  # Max 50 from objects
        texture_score = min(50, edge_score)  # Max 50 from texture
        
        total_score = object_score + texture_score
        
        # Determine level
        if total_score < 25:
            level = "clean"
        elif total_score < 50:
            level = "moderate"
        else:
            level = "messy"
        
        result = {
            "score": total_score,
            "level": level,
            "messy_objects": messy_count,
            "edge_score": edge_score,
            "total_objects": len(detections)
        }
        
        logger.info(f"Messiness analysis: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Messiness analysis error: {e}")
        return {
            "score": 0,
            "level": "clean",
            "messy_objects": 0,
            "edge_score": 0,
            "total_objects": 0
        }


def describe_room(detections: List[Dict]) -> str:
    """
    Generate a natural language description of the room.
    
    Args:
        detections: List of detected objects
        
    Returns:
        Room description text
    """
    if not detections:
        return "I no see anything. Room dey empty."
    
    # Group objects by category
    categories = {
        "furniture": [],
        "electronics": [],
        "items": [],
        "people": []
    }
    
    furniture = {'chair', 'couch', 'bed', 'dining table', 'toilet', 'sink'}
    electronics = {'laptop', 'phone', 'tv', 'computer', 'keyboard', 'mouse', 'remote'}
    people = {'person'}
    
    for det in detections:
        class_name = det['class'].lower()
        
        if class_name in furniture:
            categories["furniture"].append(class_name)
        elif class_name in electronics:
            categories["electronics"].append(class_name)
        elif class_name in people:
            categories["people"].append(class_name)
        else:
            categories["items"].append(class_name)
    
    # Build description
    parts = []
    
    if categories["people"]:
        count = len(categories["people"])
        if count == 1:
            parts.append("I see one person")
        else:
            parts.append(f"I see {count} persons")
    
    if categories["furniture"]:
        unique_furniture = list(set(categories["furniture"]))
        if len(unique_furniture) == 1:
            parts.append(f"get {unique_furniture[0]}")
        else:
            furniture_list = ", ".join(unique_furniture[:-1]) + " and " + unique_furniture[-1]
            parts.append(f"get {furniture_list}")
    
    if categories["electronics"]:
        unique_electronics = list(set(categories["electronics"]))
        if len(unique_electronics) == 1:
            parts.append(f"one {unique_electronics[0]}")
        else:
            parts.append("some electronics")
    
    if categories["items"]:
        unique_items = list(set(categories["items"]))
        if len(unique_items) <= 3:
            items_list = ", ".join(unique_items)
            parts.append(f"some {items_list}")
        else:
            parts.append(f"plenty things I no go mention")
    
    if not parts:
        return "I see some things but I no fit describe am well well."
    
    description = ", ".join(parts[:3])  # Limit to 3 main points
    return description + "."


def process_frame(frame_data: bytes) -> Dict:
    """
    Process a single frame and return analysis results.
    
    Args:
        frame_data: Image bytes from the frame
        
    Returns:
        Dictionary with detections, messiness, and description
    """
    try:
        # Detect objects
        detections, annotated_image = detect_objects(frame_data)
        
        if not detections:
            return {
                "detections": [],
                "messiness": {"score": 0, "level": "clean"},
                "description": "I no see anything clear.",
                "annotated_image": None
            }
        
        # Analyze messiness
        messiness = analyze_messiness(detections, annotated_image)
        
        # Generate description
        description = describe_room(detections)
        
        # Convert annotated image to base64
        annotated_b64 = None
        if annotated_image:
            buffered = io.BytesIO()
            annotated_image.save(buffered, format="JPEG", quality=70)
            annotated_b64 = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "detections": detections,
            "messiness": messiness,
            "description": description,
            "annotated_image": annotated_b64
        }
        
    except Exception as e:
        logger.error(f"Frame processing error: {e}")
        return {
            "detections": [],
            "messiness": {"score": 0, "level": "clean"},
            "description": "Error processing frame.",
            "annotated_image": None,
            "error": str(e)
        }


def initialize_model():
    """
    Initialize the model on startup.
    """
    try:
        get_model()
        logger.info("Vision model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize vision model: {e}")
        # Don't raise - allow app to work without vision
        logger.warning("Vision model unavailable, camera features will be disabled")
