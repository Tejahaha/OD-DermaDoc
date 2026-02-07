"""
=============================================================================
YOLOv8 SEGMENTATION PIPELINE FOR SKIN LESION ANALYSIS
=============================================================================

This script walks you through every step of training a segmentation model,
from understanding your data to inspecting model failures.

WHAT IS SEGMENTATION?
- Detection: draws a BOX around objects
- Segmentation: draws the EXACT SHAPE (polygon/mask) around objects

Your dataset uses YOLO segmentation format:
  Each label line: class_id x1 y1 x2 y2 x3 y3 ... (normalized polygon coordinates)
  
Example: "5 0.49 0.14 0.48 0.15 ..." means:
  - Class ID: 5 (a type of skin lesion)
  - Polygon points: (0.49, 0.14), (0.48, 0.15), etc.
  - Coordinates are normalized (0-1) relative to image width/height

=============================================================================
"""

# IMPORTANT: Set matplotlib backend BEFORE importing pyplot
# This avoids Tcl/Tk display issues on Windows
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (saves to files)

import os
import random
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# ============================================================================
# CONFIGURATION - Edit these paths if needed
# ============================================================================

# Base path to your dataset
DATASET_PATH = r"c:\Users\TEJA\PycharmProjects\DermaDoc\skin-lesion-segmentation-classification"

# Paths to each split
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VALID_PATH = os.path.join(DATASET_PATH, "valid")
TEST_PATH = os.path.join(DATASET_PATH, "test")

# Output directory for trained models
OUTPUT_DIR = r"c:\Users\TEJA\PycharmProjects\DermaDoc\runs"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_image_label_pairs(split_path):
    """
    Find all image-label pairs in a split folder.
    
    YOLO expects this structure:
        split/
            images/
                image1.jpg
                image2.jpg
            labels/
                image1.txt  (same name as image)
                image2.txt
    
    Returns: List of (image_path, label_path) tuples
    """
    images_dir = os.path.join(split_path, "images")
    labels_dir = os.path.join(split_path, "labels")
    
    pairs = []
    
    # Find all images
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    
    for img_path in image_files:
        # Get corresponding label file (same name, .txt extension)
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, f"{img_name}.txt")
        
        if os.path.exists(label_path):
            pairs.append((img_path, label_path))
    
    return pairs


def parse_yolo_segmentation_label(label_path, img_width, img_height):
    """
    Parse a YOLO segmentation label file.
    
    YOLO format: class_id x1 y1 x2 y2 x3 y3 ... (normalized 0-1)
    
    Returns: List of (class_id, polygon_points) where polygon_points is 
             a numpy array of shape (N, 2) with actual pixel coordinates
    """
    objects = []
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # Need at least class_id + 3 points (6 coords)
                continue
            
            class_id = int(parts[0])
            
            # Parse polygon coordinates (pairs of x, y)
            coords = list(map(float, parts[1:]))
            
            # Group into (x, y) pairs
            points = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    x = coords[i] * img_width      # Denormalize x
                    y = coords[i + 1] * img_height  # Denormalize y
                    points.append([x, y])
            
            if len(points) >= 3:  # Need at least 3 points for a polygon
                objects.append((class_id, np.array(points, dtype=np.int32)))
    
    return objects


def visualize_segmentation(image, objects, title="Segmentation Visualization"):
    """
    Overlay segmentation polygons on an image.
    
    Each object is drawn with:
    - A filled semi-transparent polygon (the mask)
    - A colored outline
    - The class ID label
    """
    # Create a copy for drawing
    vis_image = image.copy()
    
    # Color palette for different classes (BGR format for OpenCV)
    colors = [
        (255, 0, 0),      # Blue
        (0, 255, 0),      # Green
        (0, 0, 255),      # Red
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Yellow
        (128, 0, 255),    # Orange
        (255, 128, 0),    # Light Blue
    ]
    
    # Create overlay for transparent fill
    overlay = vis_image.copy()
    
    for class_id, polygon in objects:
        color = colors[class_id % len(colors)]
        
        # Draw filled polygon on overlay (for transparency)
        cv2.fillPoly(overlay, [polygon], color)
        
        # Draw polygon outline on main image
        cv2.polylines(vis_image, [polygon], isClosed=True, 
                      color=color, thickness=2)
        
        # Add class label
        centroid = polygon.mean(axis=0).astype(int)
        cv2.putText(vis_image, f"Class {class_id}", 
                    tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, color, 2)
    
    # Blend overlay with original (0.4 = 40% opacity for fill)
    vis_image = cv2.addWeighted(overlay, 0.4, vis_image, 0.6, 0)
    
    # Convert BGR to RGB for matplotlib
    vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
    
    return vis_image_rgb


# ============================================================================
# STEP 1: DATASET SANITY CHECK
# ============================================================================

def step1_dataset_sanity_check():
    """
    PURPOSE: Understand what your labels look like when overlaid on images.
    
    This helps you verify:
    1. Labels are correctly aligned with images
    2. Polygons accurately trace lesion boundaries
    3. Class IDs make sense
    """
    print("\n" + "="*70)
    print("STEP 1: DATASET SANITY CHECK")
    print("="*70)
    print("""
    WHAT WE'RE DOING:
    - Loading a random image from the training set
    - Reading its segmentation label (polygon coordinates)
    - Drawing the polygon on top of the image
    
    WHAT TO LOOK FOR:
    - The colored polygon should trace the skin lesion boundary
    - If the polygon looks wrong, there might be a labeling issue
    """)
    
    # Get all image-label pairs from training set
    pairs = get_image_label_pairs(TRAIN_PATH)
    
    if not pairs:
        print("ERROR: No image-label pairs found in training set!")
        print(f"Checked path: {TRAIN_PATH}")
        return
    
    # Pick a random pair
    img_path, label_path = random.choice(pairs)
    
    print(f"\n📁 Selected image: {os.path.basename(img_path)}")
    print(f"📄 Label file: {os.path.basename(label_path)}")
    
    # Load the image
    image = cv2.imread(img_path)
    if image is None:
        print(f"ERROR: Could not load image: {img_path}")
        return
    
    img_height, img_width = image.shape[:2]
    print(f"📐 Image dimensions: {img_width} x {img_height} pixels")
    
    # Parse the label file
    objects = parse_yolo_segmentation_label(label_path, img_width, img_height)
    print(f"🎯 Found {len(objects)} object(s) in this image")
    
    # Show raw label content
    print(f"\n📝 Raw label content (first 200 chars):")
    with open(label_path, 'r') as f:
        content = f.read()[:200]
        print(f"   '{content}...'")
    
    for i, (class_id, polygon) in enumerate(objects):
        print(f"   Object {i+1}: Class {class_id}, {len(polygon)} polygon points")
    
    # Visualize
    vis_image = visualize_segmentation(image, objects)
    
    # Display
    plt.figure(figsize=(12, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image", fontsize=14)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(vis_image)
    plt.title("With Segmentation Overlay", fontsize=14)
    plt.axis('off')
    
    plt.suptitle(f"STEP 1: Sanity Check - {os.path.basename(img_path)}", fontsize=16)
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "step1_sanity_check.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"\n📸 Image saved to: {output_path}")
    print("   Open this file to see the visualization!")
    
    print("\n✅ STEP 1 COMPLETE!")
    print("   Check that the polygon traces the lesion boundary correctly.")


# ============================================================================
# STEP 2: DATASET STATISTICS
# ============================================================================

def step2_dataset_statistics():
    """
    PURPOSE: Get a comprehensive overview of your dataset.
    
    Understanding your data helps you:
    1. Know if you have enough samples for training
    2. Identify class imbalance (some classes have more examples)
    3. Spot potential issues (empty labels, missing files)
    """
    print("\n" + "="*70)
    print("STEP 2: DATASET STATISTICS")
    print("="*70)
    print("""
    WHAT WE'RE DOING:
    - Counting images in each split (train/valid/test)
    - Counting objects per image
    - Finding all unique class IDs
    
    WHY THIS MATTERS:
    - More training images = better model (usually)
    - Balanced classes = fairer predictions
    - Knowing your classes helps interpret results
    """)
    
    splits = {
        "Train": TRAIN_PATH,
        "Valid": VALID_PATH,
        "Test": TEST_PATH
    }
    
    all_stats = {}
    all_class_ids = set()
    
    for split_name, split_path in splits.items():
        pairs = get_image_label_pairs(split_path)
        
        objects_per_image = []
        class_counts = {}
        
        for img_path, label_path in pairs:
            # Get image dimensions for parsing
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            img_height, img_width = image.shape[:2]
            objects = parse_yolo_segmentation_label(label_path, img_width, img_height)
            
            objects_per_image.append(len(objects))
            
            for class_id, _ in objects:
                all_class_ids.add(class_id)
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        all_stats[split_name] = {
            "num_images": len(pairs),
            "objects_per_image": objects_per_image,
            "class_counts": class_counts
        }
    
    # Print statistics
    print("\n📊 IMAGE COUNTS PER SPLIT:")
    print("-" * 40)
    for split_name, stats in all_stats.items():
        print(f"   {split_name}: {stats['num_images']} images")
    
    print("\n📊 OBJECTS PER IMAGE:")
    print("-" * 40)
    for split_name, stats in all_stats.items():
        if stats['objects_per_image']:
            avg = np.mean(stats['objects_per_image'])
            min_obj = np.min(stats['objects_per_image'])
            max_obj = np.max(stats['objects_per_image'])
            print(f"   {split_name}: min={min_obj}, max={max_obj}, avg={avg:.2f}")
    
    print(f"\n📊 UNIQUE CLASS IDs FOUND: {sorted(all_class_ids)}")
    print("-" * 40)
    
    # Aggregate class counts from training set
    train_class_counts = all_stats["Train"]["class_counts"]
    print("   Class distribution in training set:")
    for class_id in sorted(train_class_counts.keys()):
        count = train_class_counts[class_id]
        print(f"      Class {class_id}: {count} instances")
    
    # Visualize statistics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Images per split
    split_names = list(all_stats.keys())
    image_counts = [all_stats[s]["num_images"] for s in split_names]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    axes[0].bar(split_names, image_counts, color=colors)
    axes[0].set_title("Images per Split", fontsize=14)
    axes[0].set_ylabel("Number of Images")
    for i, v in enumerate(image_counts):
        axes[0].text(i, v + 10, str(v), ha='center', fontweight='bold')
    
    # Plot 2: Objects per image distribution (training set)
    train_objects = all_stats["Train"]["objects_per_image"]
    axes[1].hist(train_objects, bins=range(0, max(train_objects)+2), 
                 color='#9b59b6', edgecolor='white')
    axes[1].set_title("Objects per Image (Training)", fontsize=14)
    axes[1].set_xlabel("Number of Objects")
    axes[1].set_ylabel("Number of Images")
    
    # Plot 3: Class distribution
    if train_class_counts:
        classes = sorted(train_class_counts.keys())
        counts = [train_class_counts[c] for c in classes]
        axes[2].bar([f"Class {c}" for c in classes], counts, color='#f39c12')
        axes[2].set_title("Class Distribution (Training)", fontsize=14)
        axes[2].set_ylabel("Number of Instances")
        axes[2].tick_params(axis='x', rotation=45)
    
    plt.suptitle("STEP 2: Dataset Statistics", fontsize=16)
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "step2_statistics.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"\n📸 Image saved to: {output_path}")
    print("   Open this file to see the statistics charts!")
    
    print("\n✅ STEP 2 COMPLETE!")
    print("   Review the statistics to understand your dataset composition.")
    
    return all_class_ids


# ============================================================================
# STEP 3: MODEL LOADING
# ============================================================================

def step3_model_loading():
    """
    PURPOSE: Load a pretrained YOLOv8 segmentation model and understand it.
    
    WHAT IS A PRETRAINED MODEL?
    - The model has already learned to recognize objects from a huge dataset (COCO)
    - It knows concepts like: edges, textures, shapes, object boundaries
    - We'll fine-tune it to recognize YOUR specific skin lesions
    - This is called "transfer learning" - transferring knowledge to a new task
    """
    print("\n" + "="*70)
    print("STEP 3: MODEL LOADING")
    print("="*70)
    print("""
    WHAT WE'RE DOING:
    - Loading YOLOv8-seg (the segmentation variant of YOLOv8)
    - Using a pretrained model (trained on 80 object categories)
    - Examining the model architecture
    
    WHY USE A PRETRAINED MODEL?
    - Training from scratch needs MILLIONS of images
    - Pretrained models already understand basic visual concepts:
      * Edges and contours (useful for lesion boundaries!)
      * Textures (skin patterns!)
      * Color gradients (lesion coloration!)
    - We just need to teach it: "THIS is what skin lesions look like"
    """)
    
    # Import Ultralytics YOLO
    from ultralytics import YOLO
    
    # Load pretrained YOLOv8 segmentation model
    # 'yolov8n-seg' = nano (smallest, fastest, good for learning)
    # Other options: yolov8s-seg (small), yolov8m-seg (medium), yolov8l-seg (large)
    print("\n⏳ Loading YOLOv8n-seg pretrained model...")
    model = YOLO('yolov8n-seg.pt')  # Downloads automatically if not present
    
    print("\n✅ Model loaded successfully!")
    
    # Print model information
    print("\n📋 MODEL SUMMARY:")
    print("-" * 40)
    print(f"   Model type: YOLOv8 Nano Segmentation")
    print(f"   Task: Instance Segmentation")
    print(f"   Pretrained on: COCO dataset (80 classes)")
    
    # Get model info
    model_info = model.info()
    
    print("""
    🧠 WHAT THE PRETRAINED MODEL KNOWS:
    
    The model was trained on COCO dataset with 80 everyday objects:
    - People, cars, animals, furniture, food, etc.
    
    Even though it hasn't seen skin lesions, it has learned:
    
    1. EDGE DETECTION
       → Useful for finding lesion boundaries
       
    2. TEXTURE RECOGNITION  
       → Helpful for distinguishing lesion from normal skin
       
    3. COLOR PATTERNS
       → Important for identifying pigmented lesions
       
    4. SHAPE UNDERSTANDING
       → Essential for tracing irregular lesion shapes
    
    When we fine-tune on YOUR data, the model will:
    - Keep its general visual understanding
    - Learn the specific appearance of skin lesions
    - Adapt its outputs to YOUR class labels
    """)
    
    print("\n✅ STEP 3 COMPLETE!")
    print("   The pretrained model is ready for fine-tuning on your data.")
    
    return model


# ============================================================================
# STEP 4: TRAINING
# ============================================================================

def step4_training(num_classes):
    """
    PURPOSE: Train the model on your skin lesion dataset.
    
    TRAINING = Teaching the model to recognize patterns in YOUR data
    
    Each training step:
    1. Model sees a batch of images
    2. Makes predictions (probably wrong at first)
    3. Compares predictions to ground truth labels
    4. Calculates "loss" (how wrong it was)
    5. Adjusts its internal weights to reduce loss
    6. Repeat thousands of times!
    """
    print("\n" + "="*70)
    print("STEP 4: TRAINING")
    print("="*70)
    print("""
    WHAT WE'RE DOING:
    - Training for a few epochs (passes through the entire dataset)
    - Using a small number of epochs for quick demonstration
    - In real projects, you'd train for 50-300 epochs
    
    WHAT IS AN EPOCH?
    - 1 epoch = model sees every training image once
    - More epochs = more learning (up to a point)
    - Too many epochs = overfitting (memorizing instead of learning)
    
    LOSSES EXPLAINED:
    - box_loss: How accurate are the bounding boxes?
    - seg_loss: How accurate are the segmentation masks?
    - cls_loss: How accurate are the class predictions?
    - dfl_loss: Distribution focal loss (helps with precise localization)
    
    Lower loss = better model (usually)
    """)
    
    from ultralytics import YOLO
    
    # First, we need to create a data.yaml file for YOLO
    # This tells YOLO where your data is and what classes you have
    
    data_yaml_path = os.path.join(DATASET_PATH, "data.yaml")
    
    # Create the data.yaml file
    data_yaml_content = f"""# Skin Lesion Segmentation Dataset
# Auto-generated by training script

path: {DATASET_PATH}
train: train/images
val: valid/images  
test: test/images

# Number of classes
nc: {num_classes}

# Class names (adjust based on your actual classes)
names:
"""
    # Add class names
    for i in range(num_classes):
        data_yaml_content += f"  {i}: class_{i}\n"
    
    with open(data_yaml_path, 'w') as f:
        f.write(data_yaml_content)
    
    print(f"\n📄 Created data configuration: {data_yaml_path}")
    print(f"   Number of classes: {num_classes}")
    
    # Load a fresh pretrained model
    print("\n⏳ Loading pretrained model for training...")
    model = YOLO('yolov8n-seg.pt')
    
    # Training parameters
    EPOCHS = 5  # Small number for demonstration (use 50-100 for real training)
    BATCH_SIZE = 8  # Reduce if you run out of GPU memory
    IMAGE_SIZE = 640  # Standard YOLO input size
    
    print(f"""
    📋 TRAINING CONFIGURATION:
    - Epochs: {EPOCHS} (use 50-100 for real training)
    - Batch size: {BATCH_SIZE}
    - Image size: {IMAGE_SIZE}x{IMAGE_SIZE}
    - Output directory: {OUTPUT_DIR}
    """)
    
    print("\n🚀 STARTING TRAINING...")
    print("=" * 50)
    print("Watch the losses decrease as the model learns!")
    print("=" * 50 + "\n")
    
    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        project=OUTPUT_DIR,
        name="skin_lesion_seg",
        exist_ok=True,  # Overwrite existing runs
        pretrained=True,
        verbose=True,
        plots=True,  # Generate training plots
    )
    
    print("\n" + "=" * 50)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 50)
    
    # Find the best model
    best_model_path = os.path.join(OUTPUT_DIR, "skin_lesion_seg", "weights", "best.pt")
    last_model_path = os.path.join(OUTPUT_DIR, "skin_lesion_seg", "weights", "last.pt")
    
    print(f"""
    📁 SAVED MODELS:
    - Best model: {best_model_path}
    - Last model: {last_model_path}
    
    The 'best' model is the one with lowest validation loss.
    Use this for inference!
    """)
    
    print("\n✅ STEP 4 COMPLETE!")
    print("   Check the runs folder for training curves and metrics.")
    
    return best_model_path


# ============================================================================
# STEP 5: VALIDATION VISUALIZATION
# ============================================================================

def step5_validation_visualization(model_path):
    """
    PURPOSE: See how well the trained model performs on validation images.
    
    This is the moment of truth - does the model actually work?
    
    We'll:
    1. Load the trained model
    2. Run it on validation images
    3. Visualize the predicted masks
    4. Compare with ground truth (if available)
    """
    print("\n" + "="*70)
    print("STEP 5: VALIDATION VISUALIZATION")
    print("="*70)
    print("""
    WHAT WE'RE DOING:
    - Loading our trained model
    - Running inference on validation images
    - Displaying predicted segmentation masks
    
    WHAT TO LOOK FOR:
    - Do predictions cover the actual lesions?
    - Are boundaries smooth or jagged?
    - Are there any false positives (detecting non-lesions)?
    - Are there any false negatives (missing lesions)?
    """)
    
    from ultralytics import YOLO
    
    # Load the trained model
    print(f"\n⏳ Loading trained model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"WARNING: Model not found at {model_path}")
        print("Using pretrained model for demonstration...")
        model = YOLO('yolov8n-seg.pt')
    else:
        model = YOLO(model_path)
    
    print("✅ Model loaded!")
    
    # Get validation images
    pairs = get_image_label_pairs(VALID_PATH)
    
    if not pairs:
        print("WARNING: No validation images found, using training images...")
        pairs = get_image_label_pairs(TRAIN_PATH)
    
    # Select random images for visualization
    num_samples = min(6, len(pairs))
    selected_pairs = random.sample(pairs, num_samples)
    
    print(f"\n📸 Running inference on {num_samples} validation images...")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (img_path, label_path) in enumerate(selected_pairs):
        if idx >= 6:
            break
            
        # Load image
        image = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = model(img_path, verbose=False)[0]
        
        # Create visualization
        vis_image = img_rgb.copy()
        
        # Draw predictions
        if results.masks is not None:
            masks = results.masks.data.cpu().numpy()
            boxes = results.boxes.data.cpu().numpy()
            
            # Color palette
            colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255),
                (255, 255, 0), (255, 0, 255), (0, 255, 255)
            ]
            
            for i, (mask, box) in enumerate(zip(masks, boxes)):
                color = colors[i % len(colors)]
                
                # Resize mask to image size
                mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                mask_bool = mask_resized > 0.5
                
                # Create colored overlay
                overlay = vis_image.copy()
                overlay[mask_bool] = color
                
                # Blend
                vis_image = cv2.addWeighted(overlay, 0.4, vis_image, 0.6, 0)
                
                # Draw contour
                contours, _ = cv2.findContours(
                    mask_bool.astype(np.uint8), 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(vis_image, contours, -1, color, 2)
                
                # Add confidence score
                conf = box[4]
                cls = int(box[5])
                label = f"Class {cls}: {conf:.2f}"
                
                # Find top-left of mask for label placement
                y_coords, x_coords = np.where(mask_bool)
                if len(y_coords) > 0:
                    text_y = max(20, min(y_coords))
                    text_x = min(x_coords)
                    cv2.putText(vis_image, label, (text_x, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        axes[idx].imshow(vis_image)
        axes[idx].set_title(os.path.basename(img_path), fontsize=10)
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(num_samples, 6):
        axes[idx].axis('off')
    
    plt.suptitle("STEP 5: Validation Predictions\n(Colored regions = predicted lesions)", fontsize=16)
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "step5_validation_results.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"\n📸 Image saved to: {output_path}")
    print("   Open this file to see the validation predictions!")
    
    print("\n✅ STEP 5 COMPLETE!")
    print("   Review the predictions to assess model quality.")
    
    return model


# ============================================================================
# STEP 6: FAILURE INSPECTION
# ============================================================================

def step6_failure_inspection(model_path):
    """
    PURPOSE: Understand WHY the model makes mistakes.
    
    Every model makes mistakes. Understanding failures helps you:
    1. Identify data quality issues
    2. Decide if you need more training data
    3. Know when the model shouldn't be trusted
    
    COMMON FAILURE CAUSES:
    - Small lesions: Hard to detect, few pixels to learn from
    - Blurry images: Model can't see clear boundaries
    - Low contrast: Lesion looks similar to surrounding skin
    - Unusual shapes: Training data might not have similar examples
    - Hair/artifacts: Occlusions confuse the model
    """
    print("\n" + "="*70)
    print("STEP 6: FAILURE INSPECTION")
    print("="*70)
    print("""
    WHAT WE'RE DOING:
    - Running the model on various images
    - Finding cases where predictions are poor
    - Analyzing WHY failures occur
    
    COMMON FAILURE MODES IN SKIN LESION SEGMENTATION:
    
    1. SMALL LESIONS
       → Few pixels = little information for the model
       → Solution: Include more small lesion examples in training
    
    2. LOW CONTRAST
       → Lesion color similar to surrounding skin
       → Solution: Use augmentation (brightness/contrast changes)
    
    3. FUZZY BOUNDARIES
       → Gradual transition, no clear edge
       → Solution: More training data with fuzzy boundaries
    
    4. OCCLUSIONS (HAIR, RULERS, ETC.)
       → Artifacts covering parts of the lesion
       → Solution: Data augmentation or preprocessing
    
    5. UNUSUAL SHAPES
       → Very irregular shapes not seen during training
       → Solution: More diverse training examples
    """)
    
    from ultralytics import YOLO
    
    # Load model
    if os.path.exists(model_path):
        model = YOLO(model_path)
    else:
        print("Using pretrained model for demonstration...")
        model = YOLO('yolov8n-seg.pt')
    
    # Get validation images
    pairs = get_image_label_pairs(VALID_PATH)
    if not pairs:
        pairs = get_image_label_pairs(TRAIN_PATH)
    
    # Find potential failure cases
    # We'll look at images and compare prediction quality
    
    print("\n🔍 Analyzing predictions to find potential failures...")
    
    failure_cases = []
    
    for img_path, label_path in pairs[:50]:  # Check first 50 images
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        img_height, img_width = image.shape[:2]
        
        # Get ground truth
        gt_objects = parse_yolo_segmentation_label(label_path, img_width, img_height)
        
        # Get predictions
        results = model(img_path, verbose=False)[0]
        
        # Analyze potential failures
        has_gt = len(gt_objects) > 0
        has_pred = results.masks is not None and len(results.masks) > 0
        
        failure_reason = None
        
        if has_gt and not has_pred:
            failure_reason = "FALSE NEGATIVE: Model missed the lesion entirely"
        elif not has_gt and has_pred:
            failure_reason = "FALSE POSITIVE: Model detected something that isn't there"
        elif has_gt and has_pred:
            # Check prediction confidence
            conf = results.boxes.conf.cpu().numpy()
            if conf.min() < 0.3:
                failure_reason = "LOW CONFIDENCE: Model is uncertain about this prediction"
        
        if failure_reason:
            failure_cases.append({
                'img_path': img_path,
                'label_path': label_path,
                'reason': failure_reason,
                'results': results,
                'gt_objects': gt_objects
            })
    
    if not failure_cases:
        print("No obvious failures found! Showing random examples for analysis...")
        # Show some random examples anyway
        failure_cases = [{
            'img_path': p[0],
            'label_path': p[1],
            'reason': "Example for analysis",
            'results': model(p[0], verbose=False)[0],
            'gt_objects': parse_yolo_segmentation_label(
                p[1], 
                cv2.imread(p[0]).shape[1], 
                cv2.imread(p[0]).shape[0]
            )
        } for p in random.sample(pairs, min(3, len(pairs)))]
    
    # Visualize failure cases
    num_show = min(3, len(failure_cases))
    fig, axes = plt.subplots(num_show, 3, figsize=(18, 6*num_show))
    
    if num_show == 1:
        axes = axes.reshape(1, -1)
    
    for idx, case in enumerate(failure_cases[:num_show]):
        img_path = case['img_path']
        image = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Column 1: Original image
        axes[idx, 0].imshow(img_rgb)
        axes[idx, 0].set_title("Original Image", fontsize=12)
        axes[idx, 0].axis('off')
        
        # Column 2: Ground truth
        gt_vis = visualize_segmentation(image, case['gt_objects'])
        axes[idx, 1].imshow(gt_vis)
        axes[idx, 1].set_title("Ground Truth Labels", fontsize=12)
        axes[idx, 1].axis('off')
        
        # Column 3: Prediction
        pred_vis = img_rgb.copy()
        results = case['results']
        
        if results.masks is not None:
            masks = results.masks.data.cpu().numpy()
            for mask in masks:
                mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                mask_bool = mask_resized > 0.5
                overlay = pred_vis.copy()
                overlay[mask_bool] = [0, 255, 0]
                pred_vis = cv2.addWeighted(overlay, 0.4, pred_vis, 0.6, 0)
        
        axes[idx, 2].imshow(pred_vis)
        axes[idx, 2].set_title(f"Prediction\n({case['reason']})", fontsize=10)
        axes[idx, 2].axis('off')
    
    plt.suptitle("STEP 6: Failure Analysis\n(Compare ground truth vs predictions)", fontsize=16)
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "step6_failure_analysis.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"\n📸 Image saved to: {output_path}")
    print("   Open this file to see the failure analysis!")
    
    print("""
    💡 HOW TO IMPROVE YOUR MODEL:
    
    1. MORE DATA: If possible, add more training images
    
    2. DATA AUGMENTATION: Add transformations like:
       - Rotation, flipping, scaling
       - Brightness/contrast changes
       - Adding synthetic noise
    
    3. LONGER TRAINING: Try 50-100 epochs instead of 5
    
    4. LARGER MODEL: Try yolov8s-seg or yolov8m-seg
    
    5. HYPERPARAMETER TUNING: Adjust learning rate, batch size
    
    6. DATA CLEANING: Remove bad labels, fix misaligned annotations
    """)
    
    print("\n✅ STEP 6 COMPLETE!")
    print("   Use these insights to improve your model.")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Run the complete YOLOv8 segmentation pipeline.
    
    Each step builds on the previous one:
    1. Understand the data format
    2. Know your dataset statistics
    3. Load and understand the model
    4. Train the model
    5. Visualize results
    6. Learn from failures
    """
    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║   YOLOv8 SEGMENTATION PIPELINE FOR SKIN LESIONS                       ║
    ║   A beginner-friendly walkthrough                                     ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    
    This script will guide you through:
    
    📋 STEP 1: Dataset Sanity Check
              → See what your data looks like
    
    📊 STEP 2: Dataset Statistics  
              → Understand your data distribution
    
    🧠 STEP 3: Model Loading
              → Load and understand the pretrained model
    
    🎯 STEP 4: Training
              → Train the model on your data
    
    ✅ STEP 5: Validation Visualization
              → See how well the model performs
    
    🔍 STEP 6: Failure Inspection
              → Learn from model mistakes
    
    Let's begin!
    """)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📁 Output directory: {OUTPUT_DIR}\n")
    
    # =========================================================================
    # STEP 1: Dataset Sanity Check
    # =========================================================================
    step1_dataset_sanity_check()
    input("\n⏸️  Press Enter to continue to STEP 2...")
    
    # =========================================================================
    # STEP 2: Dataset Statistics
    # =========================================================================
    class_ids = step2_dataset_statistics()
    num_classes = max(class_ids) + 1 if class_ids else 1
    input("\n⏸️  Press Enter to continue to STEP 3...")
    
    # =========================================================================
    # STEP 3: Model Loading
    # =========================================================================
    step3_model_loading()
    input("\n⏸️  Press Enter to continue to STEP 4 (Training)...")
    
    # =========================================================================
    # STEP 4: Training
    # =========================================================================
    print("\n⚠️  NOTE: Training will start. This may take several minutes.")
    print("    For a real project, increase epochs to 50-100.")
    confirm = input("    Start training? (y/n): ")
    
    if confirm.lower() == 'y':
        best_model_path = step4_training(num_classes)
    else:
        print("    Skipping training. Using pretrained model for remaining steps.")
        best_model_path = 'yolov8n-seg.pt'
    
    input("\n⏸️  Press Enter to continue to STEP 5...")
    
    # =========================================================================
    # STEP 5: Validation Visualization
    # =========================================================================
    step5_validation_visualization(best_model_path)
    input("\n⏸️  Press Enter to continue to STEP 6...")
    
    # =========================================================================  
    # STEP 6: Failure Inspection
    # =========================================================================
    step6_failure_inspection(best_model_path)
    
    # =========================================================================
    # COMPLETION
    # =========================================================================
    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║   🎉 PIPELINE COMPLETE!                                               ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    
    📁 OUTPUT FILES SAVED:
    - step1_sanity_check.png    → Data visualization
    - step2_statistics.png      → Dataset statistics
    - step5_validation_results.png → Model predictions
    - step6_failure_analysis.png   → Failure cases
    - skin_lesion_seg/          → Training outputs and saved model
    
    🚀 NEXT STEPS TO IMPROVE YOUR MODEL:
    
    1. INCREASE TRAINING:
       → Change EPOCHS = 5 to EPOCHS = 50 or more
       
    2. TRY A BIGGER MODEL:
       → Change 'yolov8n-seg.pt' to 'yolov8s-seg.pt' or 'yolov8m-seg.pt'
       
    3. ADD DATA AUGMENTATION:
       → Add parameters like: augment=True, mosaic=1.0, mixup=0.1
       
    4. FINE-TUNE HYPERPARAMETERS:
       → Adjust learning rate: lr0=0.01
       → Adjust batch size based on your GPU memory
       
    5. COLLECT MORE DATA:
       → More diverse examples = better generalization
    
    Happy learning! 🎓
    """)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
