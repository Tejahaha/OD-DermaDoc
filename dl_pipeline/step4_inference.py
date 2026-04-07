"""
=============================================================================
STEP 4: INFERENCE ON NEW IMAGES
=============================================================================

Run the trained YOLOv8 skin-lesion segmentation model on arbitrary
images that were never seen during training or validation.

What this module does
---------------------
``run_inference``
    Feeds a single image to the model and returns a structured dictionary
    containing every detected lesion's class, confidence, bounding box,
    severity level, and raw mask shape.

``visualize_inference``
    Renders the detections as severity-colour-coded overlays:
    * Green  – Benign
    * Orange – Pre-cancerous
    * Red    – Malignant (requires urgent clinical attention)
    Each lesion is labelled with its name, severity text, and confidence
    percentage.  A summary banner counts total detections.

``process_single_image``
    High-level wrapper: loads the model, calls ``run_inference``,
    saves the annotated JPEG, and optionally writes a JSON sidecar.

``process_folder``
    Iterates over all supported images in a directory and calls
    ``process_single_image`` for each one.

Output artefacts per image
--------------------------
* ``<base_name>_segmented.jpg`` – annotated image with overlays.
* ``<base_name>_results.json``  – machine-readable detection records.

Command-line usage
------------------
    # Single image
    python step4_inference.py --image path/to/image.jpg

    # Folder of images
    python step4_inference.py --folder path/to/images/

    # Custom model and output directory
    python step4_inference.py --image image.jpg --model weights/best.pt --output out/

    # Skip JSON output
    python step4_inference.py --image image.jpg --no-json
"""

import os
import sys
import argparse
import glob
import json
from datetime import datetime
from typing import Dict, List, Optional

# Make the parent package importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from dl_pipeline.config import (
    OUTPUT_DIR, IMAGE_EXTENSIONS, CLASS_NAMES, CLASS_CODES,
    CLASS_COLORS, CLASS_SEVERITY, CONF_THRESHOLD, IOU_THRESHOLD
)


# ============================================================================
# MODEL UTILITIES
# ============================================================================

def get_default_model_path() -> str:
    """
    Return the expected path to the best model weights from Step 2.

    Returns
    -------
    str
        Absolute path to ``<OUTPUT_DIR>/yolov8_seg/weights/best.pt``.
    """
    return os.path.join(OUTPUT_DIR, "yolov8_seg", "weights", "best.pt")


def load_model(model_path: Optional[str] = None):
    """
    Load a trained YOLOv8 model from disk.

    Parameters
    ----------
    model_path : str, optional
        Path to the ``.pt`` weights file.  Defaults to
        ``<OUTPUT_DIR>/yolov8_seg/weights/best.pt`` when not provided.

    Returns
    -------
    ultralytics.YOLO or None
        Loaded model, or ``None`` if the file does not exist.
    """
    from ultralytics import YOLO

    if model_path is None:
        model_path = get_default_model_path()

    if not os.path.exists(model_path):
        print(f"  ❌ Model not found: {model_path}")
        return None

    return YOLO(model_path)


# ============================================================================
# INFERENCE
# ============================================================================

def run_inference(
    model,
    image_path: str,
    conf_threshold: Optional[float] = None,
    iou_threshold: Optional[float] = None,
) -> Dict:
    """
    Run the segmentation model on a single image and return structured results.

    Calls the model with the configured confidence and IoU thresholds,
    then converts the raw Ultralytics output into a plain Python dictionary
    that is easy to serialise as JSON or forward to the back-end API.

    The ``mask_shape`` field stores the mask tensor dimensions for reference;
    the full pixel mask is not included in the dictionary to keep it
    JSON-serialisable.

    Parameters
    ----------
    model : ultralytics.YOLO
        Loaded YOLO model (from :func:`load_model`).
    image_path : str
        Absolute or relative path to the image file.
    conf_threshold : float, optional
        Minimum confidence to retain a detection.
        Defaults to ``config.CONF_THRESHOLD`` (0.25).
    iou_threshold : float, optional
        IoU threshold for Non-Maximum Suppression.
        Defaults to ``config.IOU_THRESHOLD`` (0.45).

    Returns
    -------
    dict
        Keys:

        * ``"image_path"``      – original file path.
        * ``"image_name"``      – basename only.
        * ``"num_detections"``  – total lesions detected.
        * ``"detections"``      – list of per-lesion dicts (see below).
        * ``"timestamp"``       – ISO-8601 inference timestamp.

        Each entry in ``"detections"`` contains:

        * ``"id"``          – zero-based detection index.
        * ``"class_id"``    – integer YOLO class index.
        * ``"class_code"``  – short diagnostic code (e.g. ``"MEL"``).
        * ``"class_name"``  – full name (e.g. ``"Melanoma"``).
        * ``"severity"``    – 0 = Benign, 1 = Pre-cancerous, 2 = Malignant.
        * ``"confidence"``  – detection confidence in [0, 1].
        * ``"bbox"``        – dict with ``x1, y1, x2, y2, width, height``.
        * ``"mask_shape"``  – tuple of the raw mask tensor dimensions.
    """
    if conf_threshold is None:
        conf_threshold = CONF_THRESHOLD
    if iou_threshold is None:
        iou_threshold = IOU_THRESHOLD

    # Run inference; index [0] extracts the result for the single input image.
    results = model(
        image_path,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )[0]

    # Parse raw tensor outputs into plain Python structures.
    detections: List[Dict] = []

    if results.boxes is not None and results.masks is not None:
        boxes = results.boxes.data.cpu().numpy()   # Shape: (N, 6)
        masks = results.masks.data.cpu().numpy()   # Shape: (N, H, W)

        for i, (box, mask) in enumerate(zip(boxes, masks)):
            x1, y1, x2, y2, conf, class_id = box
            class_id = int(class_id)

            detection = {
                "id":         i,
                "class_id":   class_id,
                "class_code": CLASS_CODES.get(class_id,   f"UNK_{class_id}"),
                "class_name": CLASS_NAMES.get(class_id,   f"Unknown {class_id}"),
                "severity":   CLASS_SEVERITY.get(class_id, 0),
                "confidence": float(conf),
                "bbox": {
                    "x1":     float(x1),
                    "y1":     float(y1),
                    "x2":     float(x2),
                    "y2":     float(y2),
                    "width":  float(x2 - x1),
                    "height": float(y2 - y1),
                },
                # Store shape only; full mask arrays are not JSON-serialisable.
                "mask_shape": mask.shape,
            }
            detections.append(detection)

    return {
        "image_path":      image_path,
        "image_name":      os.path.basename(image_path),
        "num_detections":  len(detections),
        "detections":      detections,
        "timestamp":       datetime.now().isoformat(),
    }


# ============================================================================
# VISUALISATION
# ============================================================================

def visualize_inference(
    image_path: str,
    results_dict: Dict,
    output_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Render segmentation masks with severity-aware colouring and save to disk.

    Runs the model a second time to obtain the full mask tensors (which are
    not stored in ``results_dict`` to keep it JSON-friendly), then:

    * Colours each mask according to the lesion's severity:
      - Green  (0, 255, 0)   → Benign
      - Orange (0, 165, 255) → Pre-cancerous
      - Red    (0, 0, 255)   → Malignant
    * Draws a 3-pixel solid contour around each lesion.
    * Places a labelled banner with class name, severity text, and confidence.
    * Adds a summary bar at the top of the image counting total detections.
    * Saves the result to ``output_path`` when provided.

    Parameters
    ----------
    image_path : str
        Path to the original image file.
    results_dict : dict
        Output from :func:`run_inference` for this image (used for the
        detection count summary only).
    output_path : str, optional
        File path where the annotated image should be written.
        When ``None``, the image is returned but not saved.

    Returns
    -------
    numpy.ndarray or None
        Annotated image in **RGB** format, or ``None`` if the file could
        not be read.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    vis_image = image.copy()
    overlay   = vis_image.copy()   # Separate layer for alpha-blended fills.

    # Re-run inference to retrieve full mask tensors for visualisation.
    model = load_model()
    if model:
        results = model(image_path, conf=CONF_THRESHOLD, verbose=False)[0]

        if results.masks is not None:
            masks = results.masks.data.cpu().numpy()
            boxes = results.boxes.data.cpu().numpy()

            for mask, box in zip(masks, boxes):
                conf     = box[4]
                class_id = int(box[5])
                severity = CLASS_SEVERITY.get(class_id, 0)

                # Map severity level to a traffic-light colour scheme (BGR).
                if severity == 2:       # Malignant → red
                    color = (0, 0, 255)
                elif severity == 1:     # Pre-cancerous → orange
                    color = (0, 165, 255)
                else:                   # Benign → green
                    color = (0, 255, 0)

                # Upscale the mask to the original image resolution.
                mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                mask_bool    = mask_resized > 0.5

                # Fill the mask area on the overlay layer.
                overlay[mask_bool] = color

                # Draw a solid 3-pixel boundary contour.
                contours, _ = cv2.findContours(
                    mask_bool.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(vis_image, contours, -1, color, 3)

                # Compose the label string: name + severity text + confidence.
                class_name    = CLASS_NAMES.get(class_id, f"Class {class_id}")
                severity_text = ["Benign", "Pre-cancerous", "Malignant"][severity]
                label         = f"{class_name} ({severity_text}) {conf:.0%}"

                y_coords, x_coords = np.where(mask_bool)
                if len(y_coords) > 0:
                    # Place label slightly above the topmost mask pixel.
                    text_y = max(30, min(y_coords) - 10)
                    text_x = min(x_coords)

                    # Draw a filled rectangle as label background for legibility.
                    (w_txt, h_txt), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(vis_image,
                                  (text_x, text_y - h_txt - 5),
                                  (text_x + w_txt, text_y + 5),
                                  color, -1)
                    cv2.putText(vis_image, label, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Alpha-blend: 40 % filled colour + 60 % original image details.
    vis_image = cv2.addWeighted(overlay, 0.4, vis_image, 0.6, 0)

    # Add a dark banner at the very top showing the total detection count.
    num_detections = results_dict["num_detections"]
    summary        = f"Detected {num_detections} lesion(s)"
    cv2.rectangle(vis_image, (0, 0), (300, 40), (0, 0, 0), -1)
    cv2.putText(vis_image, summary, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Persist the annotated image when an output path is specified.
    if output_path:
        cv2.imwrite(output_path, vis_image)
        print(f"  ✅ Saved: {output_path}")

    # Convert BGR → RGB for Matplotlib compatibility.
    return cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)


# ============================================================================
# HIGH-LEVEL IMAGE PROCESSORS
# ============================================================================

def process_single_image(
    image_path: str,
    model_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    save_json: bool = True,
) -> Optional[Dict]:
    """
    Run inference on one image, save the annotated output, and return results.

    Combines :func:`load_model`, :func:`run_inference`,
    :func:`visualize_inference`, and optional JSON serialisation into a
    single convenient call.

    Parameters
    ----------
    image_path : str
        Path to the image file to process.
    model_path : str, optional
        Path to model weights.  Defaults to ``best.pt`` from Step 2.
    output_dir : str, optional
        Directory where ``*_segmented.jpg`` and ``*_results.json`` are saved.
        Defaults to ``config.OUTPUT_DIR``.
    save_json : bool, optional
        When ``True`` (default), write a JSON sidecar with full detection
        metadata alongside the annotated image.

    Returns
    -------
    dict or None
        The results dictionary from :func:`run_inference`, or ``None`` if
        the model could not be loaded.
    """
    print(f"\n  📸 Processing: {os.path.basename(image_path)}")

    model = load_model(model_path)
    if model is None:
        return None

    # Run inference and collect structured results.
    results = run_inference(model, image_path)

    print(f"     Found {results['num_detections']} lesion(s)")

    # Print a per-lesion summary with severity annotation.
    for det in results['detections']:
        severity_text = ["Benign", "Pre-cancerous", "⚠️ MALIGNANT"][det['severity']]
        print(f"       • {det['class_name']}: {det['confidence']:.1%} ({severity_text})")

    # Resolve the output directory.
    if output_dir is None:
        output_dir = OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)

    # Save the annotated image.
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    vis_path  = os.path.join(output_dir, f"{base_name}_segmented.jpg")
    visualize_inference(image_path, results, vis_path)

    # Optionally write a JSON sidecar (mask_shape tuple must be serialised as list).
    if save_json:
        json_path = os.path.join(output_dir, f"{base_name}_results.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=list)
        print(f"  ✅ JSON: {json_path}")

    return results


def process_folder(
    folder_path: str,
    model_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> List[Dict]:
    """
    Process all supported images in a directory.

    Discovers every file whose extension matches ``config.IMAGE_EXTENSIONS``
    and calls :func:`process_single_image` for each one.

    Parameters
    ----------
    folder_path : str
        Path to the directory containing the images to process.
    model_path : str, optional
        Path to model weights.  Defaults to ``best.pt`` from Step 2.
    output_dir : str, optional
        Directory for output artefacts.  Defaults to ``config.OUTPUT_DIR``.

    Returns
    -------
    list of dict
        One results dictionary per successfully processed image.
        Failures are silently excluded.
    """
    print(f"\n  📁 Processing folder: {folder_path}")

    # Collect all images regardless of extension case.
    image_files: List[str] = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))

    if not image_files:
        print(f"  ❌ No images found in {folder_path}")
        return []

    print(f"  📸 Found {len(image_files)} images")

    all_results: List[Dict] = []
    for img_path in image_files:
        result = process_single_image(img_path, model_path, output_dir)
        if result:
            all_results.append(result)

    return all_results


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def run_inference_cli() -> None:
    """
    Parse command-line arguments and dispatch to the appropriate processor.

    Requires exactly one of ``--image`` or ``--folder`` to be specified.
    Prints usage examples when neither is provided.

    CLI flags
    ---------
    --image str    Path to a single image file.
    --folder str   Path to a directory of images.
    --model str    Path to model weights (optional).
    --output str   Output directory for artefacts (optional).
    --no-json      Skip JSON output files.
    """
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " STEP 4: INFERENCE ON NEW IMAGES ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    parser = argparse.ArgumentParser(description="Run YOLOv8 Skin Lesion Segmentation")
    parser.add_argument('--image',   type=str, help='Path to single image')
    parser.add_argument('--folder',  type=str, help='Path to folder of images')
    parser.add_argument('--model',   type=str, default=None, help='Path to model weights')
    parser.add_argument('--output',  type=str, default=None, help='Output directory')
    parser.add_argument('--no-json', action='store_true',    help='Skip JSON output')
    args = parser.parse_args()

    # Require at least one input source.
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
