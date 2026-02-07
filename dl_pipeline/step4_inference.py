"""
=============================================================================
STEP 4: INFERENCE ON NEW IMAGES
=============================================================================

Run the trained model on new/unseen images for skin lesion segmentation.
Can process single images or entire folders.

Usage:
    # Single image
    python step4_inference.py --image path/to/image.jpg
    
    # Folder of images
    python step4_inference.py --folder path/to/images/
    
    # Save results to specific location
    python step4_inference.py --image image.jpg --output results/
"""

import os
import sys
import argparse
import glob
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from dl_pipeline.config import (
    OUTPUT_DIR, IMAGE_EXTENSIONS, CLASS_NAMES, CLASS_CODES,
    CLASS_COLORS, CLASS_SEVERITY, CONF_THRESHOLD, IOU_THRESHOLD
)


def get_default_model_path():
    """Get the default trained model path."""
    return os.path.join(OUTPUT_DIR, "yolov8_seg", "weights", "best.pt")


def load_model(model_path=None):
    """Load the trained YOLOv8 model."""
    from ultralytics import YOLO
    
    if model_path is None:
        model_path = get_default_model_path()
    
    if not os.path.exists(model_path):
        print(f"  ❌ Model not found: {model_path}")
        return None
    
    return YOLO(model_path)


def run_inference(model, image_path, conf_threshold=None, iou_threshold=None):
    """
    Run inference on a single image.
    
    Args:
        model: Loaded YOLO model
        image_path: Path to the image
        conf_threshold: Confidence threshold (default from config)
        iou_threshold: IoU threshold for NMS (default from config)
    
    Returns:
        dict with detection results
    """
    if conf_threshold is None:
        conf_threshold = CONF_THRESHOLD
    if iou_threshold is None:
        iou_threshold = IOU_THRESHOLD
    
    # Run inference
    results = model(
        image_path,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )[0]
    
    # Parse results
    detections = []
    
    if results.boxes is not None and results.masks is not None:
        boxes = results.boxes.data.cpu().numpy()
        masks = results.masks.data.cpu().numpy()
        
        for i, (box, mask) in enumerate(zip(boxes, masks)):
            x1, y1, x2, y2, conf, class_id = box
            class_id = int(class_id)
            
            detection = {
                "id": i,
                "class_id": class_id,
                "class_code": CLASS_CODES.get(class_id, f"UNK_{class_id}"),
                "class_name": CLASS_NAMES.get(class_id, f"Unknown {class_id}"),
                "severity": CLASS_SEVERITY.get(class_id, 0),
                "confidence": float(conf),
                "bbox": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "width": float(x2 - x1),
                    "height": float(y2 - y1)
                },
                "mask_shape": mask.shape
            }
            detections.append(detection)
    
    return {
        "image_path": image_path,
        "image_name": os.path.basename(image_path),
        "num_detections": len(detections),
        "detections": detections,
        "timestamp": datetime.now().isoformat()
    }


def visualize_inference(image_path, results_dict, output_path=None):
    """
    Create visualization of inference results.
    
    Args:
        image_path: Path to original image
        results_dict: Results from run_inference
        output_path: Where to save visualization (None = don't save)
    
    Returns:
        Visualization image (RGB numpy array)
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    vis_image = image.copy()
    overlay = vis_image.copy()
    
    # Load model to get masks (results_dict doesn't store full masks)
    model = load_model()
    if model:
        results = model(image_path, conf=CONF_THRESHOLD, verbose=False)[0]
        
        if results.masks is not None:
            masks = results.masks.data.cpu().numpy()
            boxes = results.boxes.data.cpu().numpy()
            
            for mask, box in zip(masks, boxes):
                conf = box[4]
                class_id = int(box[5])
                severity = CLASS_SEVERITY.get(class_id, 0)
                
                # Color based on severity
                if severity == 2:  # Malignant
                    color = (0, 0, 255)  # Red
                elif severity == 1:  # Pre-cancerous
                    color = (0, 165, 255)  # Orange
                else:  # Benign
                    color = (0, 255, 0)  # Green
                
                # Resize and apply mask
                mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                mask_bool = mask_resized > 0.5
                
                overlay[mask_bool] = color
                
                # Draw contour
                contours, _ = cv2.findContours(
                    mask_bool.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(vis_image, contours, -1, color, 3)
                
                # Add label with severity
                class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
                severity_text = ["Benign", "Pre-cancerous", "Malignant"][severity]
                label = f"{class_name} ({severity_text}) {conf:.0%}"
                
                y_coords, x_coords = np.where(mask_bool)
                if len(y_coords) > 0:
                    text_y = max(30, min(y_coords) - 10)
                    text_x = min(x_coords)
                    
                    # Draw label background
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(vis_image, (text_x, text_y - h - 5),
                                (text_x + w, text_y + 5), color, -1)
                    cv2.putText(vis_image, label, (text_x, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Blend overlay
    vis_image = cv2.addWeighted(overlay, 0.4, vis_image, 0.6, 0)
    
    # Add summary at top
    num_detections = results_dict["num_detections"]
    summary = f"Detected {num_detections} lesion(s)"
    cv2.rectangle(vis_image, (0, 0), (300, 40), (0, 0, 0), -1)
    cv2.putText(vis_image, summary, (10, 28),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add legend
    legend_y = image.shape[0] - 80
    cv2.rectangle(vis_image, (0, legend_y), (180, image.shape[0]), (0, 0, 0), -1)
    cv2.putText(vis_image, "Legend:", (10, legend_y + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.circle(vis_image, (20, legend_y + 40), 8, (0, 255, 0), -1)
    cv2.putText(vis_image, "Benign", (35, legend_y + 45),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.circle(vis_image, (100, legend_y + 40), 8, (0, 165, 255), -1)
    cv2.putText(vis_image, "Pre-cancer", (115, legend_y + 45),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.circle(vis_image, (20, legend_y + 60), 8, (0, 0, 255), -1)
    cv2.putText(vis_image, "Malignant", (35, legend_y + 65),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Save if output path provided
    if output_path:
        cv2.imwrite(output_path, vis_image)
        print(f"  ✅ Saved: {output_path}")
    
    return cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)


def process_single_image(image_path, model_path=None, output_dir=None, save_json=True):
    """Process a single image and return results."""
    print(f"\n  📸 Processing: {os.path.basename(image_path)}")
    
    model = load_model(model_path)
    if model is None:
        return None
    
    # Run inference
    results = run_inference(model, image_path)
    
    print(f"     Found {results['num_detections']} lesion(s)")
    
    for det in results['detections']:
        severity_text = ["Benign", "Pre-cancerous", "⚠️ MALIGNANT"][det['severity']]
        print(f"       • {det['class_name']}: {det['confidence']:.1%} ({severity_text})")
    
    # Save visualization
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    vis_path = os.path.join(output_dir, f"{base_name}_segmented.jpg")
    visualize_inference(image_path, results, vis_path)
    
    # Save JSON
    if save_json:
        json_path = os.path.join(output_dir, f"{base_name}_results.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  ✅ JSON: {json_path}")
    
    return results


def process_folder(folder_path, model_path=None, output_dir=None):
    """Process all images in a folder."""
    print(f"\n  📁 Processing folder: {folder_path}")
    
    # Find all images
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
    
    if not image_files:
        print(f"  ❌ No images found in {folder_path}")
        return []
    
    print(f"  📸 Found {len(image_files)} images")
    
    all_results = []
    for img_path in image_files:
        result = process_single_image(img_path, model_path, output_dir)
        if result:
            all_results.append(result)
    
    return all_results


def run_inference_cli():
    """Run inference from command line."""
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " STEP 4: INFERENCE ON NEW IMAGES ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    
    parser = argparse.ArgumentParser(description="Run YOLOv8 Skin Lesion Segmentation")
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--folder', type=str, help='Path to folder of images')
    parser.add_argument('--model', type=str, default=None, help='Path to model weights')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--no-json', action='store_true', help='Skip JSON output')
    args = parser.parse_args()
    
    if not args.image and not args.folder:
        print("  ❌ Please provide --image or --folder argument")
        print("\n  Examples:")
        print("    python step4_inference.py --image sample.jpg")
        print("    python step4_inference.py --folder images/")
        return
    
    if args.image:
        if not os.path.exists(args.image):
            print(f"  ❌ Image not found: {args.image}")
            return
        process_single_image(args.image, args.model, args.output, not args.no_json)
    
    if args.folder:
        if not os.path.isdir(args.folder):
            print(f"  ❌ Folder not found: {args.folder}")
            return
        process_folder(args.folder, args.model, args.output)
    
    print("\n" + "=" * 60)
    print("✅ STEP 4 COMPLETE!")
    print("=" * 60 + "\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_inference_cli()
