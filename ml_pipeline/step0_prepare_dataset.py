"""
=============================================================================
STEP 0: PREPARE BALANCED DATASET WITH DATA AUGMENTATION
=============================================================================

This module handles:
1. Combining all images from existing train/valid/test folders
2. Creating a balanced 70/20/10 split for train/valid/test
3. Applying data augmentation to ensure image variety

Key Functions:
    - prepare_balanced_dataset(): Main function to create balanced splits
    - get_class_distribution(): Analyze class distribution
    - apply_augmentation(): Apply random transformations to images

Usage:
    from step0_prepare_dataset import prepare_balanced_dataset
    
    prepare_balanced_dataset()  # Creates balanced_dataset/ folder
"""

import os
import glob
import shutil
import random
import cv2
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm

# Import configuration
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATASET_PATH, OUTPUT_DIR, IMAGE_EXTENSIONS, RANDOM_SEED
)

# Set random seed for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Split ratios
TRAIN_RATIO = 0.70
VALID_RATIO = 0.20
TEST_RATIO = 0.10

# Output directory for balanced dataset
BALANCED_DATASET_PATH = os.path.join(
    os.path.dirname(DATASET_PATH), 
    "balanced_dataset"
)

# Augmentation settings
AUGMENTATION_ENABLED = True


# ============================================================================
# DATA AUGMENTATION FUNCTIONS
# ============================================================================

def random_rotation(image, max_angle=30):
    """Rotate image by a random angle."""
    angle = random.uniform(-max_angle, max_angle)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def random_flip(image):
    """Randomly flip image horizontally and/or vertically."""
    if random.random() > 0.5:
        image = cv2.flip(image, 1)  # Horizontal flip
    if random.random() > 0.5:
        image = cv2.flip(image, 0)  # Vertical flip
    return image


def random_brightness_contrast(image, brightness_range=0.2, contrast_range=0.2):
    """Adjust brightness and contrast randomly."""
    alpha = 1.0 + random.uniform(-contrast_range, contrast_range)  # Contrast
    beta = random.uniform(-brightness_range, brightness_range) * 255  # Brightness
    
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted


def random_gaussian_blur(image, max_kernel=5):
    """Apply random Gaussian blur."""
    if random.random() > 0.5:
        kernel_size = random.choice([3, 5])
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return image


def random_noise(image, noise_level=0.02):
    """Add random Gaussian noise."""
    if random.random() > 0.5:
        noise = np.random.normal(0, noise_level * 255, image.shape).astype(np.float32)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy
    return image


def random_color_shift(image, shift_range=20):
    """Randomly shift color channels."""
    if random.random() > 0.5:
        shifts = np.random.randint(-shift_range, shift_range, size=3)
        shifted = image.astype(np.int32)
        for i, shift in enumerate(shifts):
            shifted[:, :, i] = np.clip(shifted[:, :, i] + shift, 0, 255)
        return shifted.astype(np.uint8)
    return image


def random_crop_resize(image, min_crop_ratio=0.85):
    """Randomly crop and resize back to original size."""
    if random.random() > 0.5:
        h, w = image.shape[:2]
        crop_ratio = random.uniform(min_crop_ratio, 1.0)
        new_h, new_w = int(h * crop_ratio), int(w * crop_ratio)
        
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)
        
        cropped = image[top:top+new_h, left:left+new_w]
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        return resized
    return image


def random_shear(image, shear_range=0.1):
    """Apply random shear transformation."""
    if random.random() > 0.5:
        h, w = image.shape[:2]
        shear_x = random.uniform(-shear_range, shear_range)
        shear_y = random.uniform(-shear_range, shear_range)
        
        M = np.array([[1, shear_x, 0],
                      [shear_y, 1, 0]], dtype=np.float32)
        return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return image


def apply_augmentation(image, intensity='medium'):
    """
    Apply a random combination of augmentations to an image.
    
    Parameters:
        image: Input image (numpy array, BGR format)
        intensity: 'light', 'medium', or 'heavy' augmentation
    
    Returns:
        Augmented image
    """
    augmented = image.copy()
    
    if intensity == 'light':
        # Light augmentation - only geometric transforms
        augmented = random_flip(augmented)
        augmented = random_rotation(augmented, max_angle=15)
        
    elif intensity == 'medium':
        # Medium augmentation - geometric + color
        augmented = random_flip(augmented)
        augmented = random_rotation(augmented, max_angle=25)
        augmented = random_brightness_contrast(augmented, 0.15, 0.15)
        augmented = random_crop_resize(augmented, 0.9)
        
    else:  # heavy
        # Heavy augmentation - all transforms
        augmented = random_flip(augmented)
        augmented = random_rotation(augmented, max_angle=30)
        augmented = random_brightness_contrast(augmented, 0.2, 0.2)
        augmented = random_gaussian_blur(augmented)
        augmented = random_noise(augmented)
        augmented = random_color_shift(augmented)
        augmented = random_crop_resize(augmented, 0.85)
        augmented = random_shear(augmented)
    
    return augmented


# ============================================================================
# DATA COLLECTION FUNCTIONS
# ============================================================================

def collect_all_samples(dataset_path):
    """
    Collect all image-label pairs from all splits (train, valid, test).
    
    Returns:
        samples_by_class: Dictionary mapping class_id to list of (image_path, label_path) tuples
    """
    samples_by_class = defaultdict(list)
    splits = ['train', 'valid', 'test']
    
    print("\n📂 Collecting all samples from dataset...")
    
    for split in splits:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            print(f"   ⚠️  Split '{split}' not found, skipping...")
            continue
            
        images_dir = os.path.join(split_path, "images")
        labels_dir = os.path.join(split_path, "labels")
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"   ⚠️  Missing images/labels folder in '{split}', skipping...")
            continue
        
        # Find all image files
        image_files = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(glob.glob(os.path.join(images_dir, ext)))
        
        print(f"   Found {len(image_files)} images in {split}/")
        
        for img_path in image_files:
            # Get corresponding label file
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(labels_dir, f"{img_name}.txt")
            
            if not os.path.exists(label_path):
                continue
            
            # Parse label to get class ID
            try:
                with open(label_path, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line:
                        class_id = int(first_line.split()[0])
                        samples_by_class[class_id].append((img_path, label_path))
            except:
                continue
    
    # Print summary
    print("\n📊 Samples collected by class:")
    total = 0
    for class_id in sorted(samples_by_class.keys()):
        count = len(samples_by_class[class_id])
        total += count
        print(f"   Class {class_id}: {count} samples")
    print(f"   Total: {total} samples")
    
    return samples_by_class


def create_balanced_splits(samples_by_class, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1):
    """
    Create balanced train/valid/test splits.
    
    Each split will have equal representation from each class (balanced).
    """
    train_samples = []
    valid_samples = []
    test_samples = []
    
    print("\n⚖️  Creating balanced splits (70/20/10)...")
    
    for class_id in sorted(samples_by_class.keys()):
        samples = samples_by_class[class_id].copy()
        random.shuffle(samples)
        
        n = len(samples)
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)
        # n_test is the remainder
        
        train_samples.extend([(s[0], s[1], class_id) for s in samples[:n_train]])
        valid_samples.extend([(s[0], s[1], class_id) for s in samples[n_train:n_train+n_valid]])
        test_samples.extend([(s[0], s[1], class_id) for s in samples[n_train+n_valid:]])
        
        print(f"   Class {class_id}: {n_train} train, {n_valid} valid, {n - n_train - n_valid} test")
    
    # Shuffle within each split
    random.shuffle(train_samples)
    random.shuffle(valid_samples)
    random.shuffle(test_samples)
    
    print(f"\n   Total: {len(train_samples)} train, {len(valid_samples)} valid, {len(test_samples)} test")
    
    return train_samples, valid_samples, test_samples


# ============================================================================
# FILE OPERATIONS
# ============================================================================

def copy_and_augment_samples(samples, output_dir, split_name, apply_aug=True):
    """
    Copy samples to output directory with optional augmentation.
    
    Parameters:
        samples: List of (image_path, label_path, class_id) tuples
        output_dir: Base output directory
        split_name: 'train', 'valid', or 'test'
        apply_aug: Whether to apply augmentation (typically only for training set)
    """
    images_out = os.path.join(output_dir, split_name, "images")
    labels_out = os.path.join(output_dir, split_name, "labels")
    
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)
    
    print(f"\n📁 Processing {split_name} set ({len(samples)} samples)...")
    
    augmentation_intensities = ['light', 'medium', 'heavy']
    
    for idx, (img_path, label_path, class_id) in enumerate(tqdm(samples, desc=f"   {split_name}")):
        # Generate unique filename
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        ext = os.path.splitext(img_path)[1]
        
        new_img_name = f"{split_name}_{idx:05d}_{base_name}{ext}"
        new_label_name = f"{split_name}_{idx:05d}_{base_name}.txt"
        
        new_img_path = os.path.join(images_out, new_img_name)
        new_label_path = os.path.join(labels_out, new_label_name)
        
        if apply_aug and AUGMENTATION_ENABLED:
            # Read image, apply augmentation, and save
            image = cv2.imread(img_path)
            if image is not None:
                # Choose random augmentation intensity
                intensity = random.choice(augmentation_intensities)
                augmented = apply_augmentation(image, intensity)
                cv2.imwrite(new_img_path, augmented)
            else:
                # Fallback to copy if image can't be read
                shutil.copy2(img_path, new_img_path)
        else:
            # Just copy without augmentation (for valid/test sets)
            shutil.copy2(img_path, new_img_path)
        
        # Copy label file (updating filename)
        shutil.copy2(label_path, new_label_path)
    
    print(f"   ✅ Saved {len(samples)} samples to {split_name}/")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def prepare_balanced_dataset(
    source_path=None, 
    output_path=None,
    train_ratio=TRAIN_RATIO,
    valid_ratio=VALID_RATIO,
    test_ratio=TEST_RATIO,
    augment_train=True
):
    """
    Main function to prepare a balanced dataset with augmentation.
    
    Parameters:
        source_path: Path to source dataset (contains train/valid/test folders)
        output_path: Path for output balanced dataset
        train_ratio: Fraction for training set (default 0.7)
        valid_ratio: Fraction for validation set (default 0.2)
        test_ratio: Fraction for test set (default 0.1)
        augment_train: Whether to apply augmentation to training images
    
    Returns:
        output_path: Path to the created balanced dataset
    """
    if source_path is None:
        source_path = DATASET_PATH
    if output_path is None:
        output_path = BALANCED_DATASET_PATH
    
    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║   📊 BALANCED DATASET PREPARATION                                     ║
    ║                                                                       ║
    ║   Creating a balanced 70/20/10 split with data augmentation           ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"   Source dataset: {source_path}")
    print(f"   Output path:    {output_path}")
    print(f"   Split ratios:   Train={train_ratio:.0%}, Valid={valid_ratio:.0%}, Test={test_ratio:.0%}")
    print(f"   Augmentation:   {'Enabled' if augment_train else 'Disabled'}")
    
    # Verify ratios sum to 1
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 0.01, \
        "Split ratios must sum to 1.0"
    
    # Step 1: Collect all samples
    samples_by_class = collect_all_samples(source_path)
    
    if not samples_by_class:
        print("\n❌ No samples found! Check your dataset path.")
        return None
    
    # Step 2: Create balanced splits
    train_samples, valid_samples, test_samples = create_balanced_splits(
        samples_by_class, train_ratio, valid_ratio, test_ratio
    )
    
    # Step 3: Create output directory
    if os.path.exists(output_path):
        print(f"\n⚠️  Output directory exists. Removing old data...")
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    
    # Step 4: Copy and augment samples
    print("\n🔄 Copying and augmenting samples...")
    
    # Training set - with augmentation
    copy_and_augment_samples(
        train_samples, output_path, "train", 
        apply_aug=augment_train
    )
    
    # Validation set - no augmentation for fair evaluation
    copy_and_augment_samples(
        valid_samples, output_path, "valid", 
        apply_aug=False
    )
    
    # Test set - no augmentation for fair evaluation
    copy_and_augment_samples(
        test_samples, output_path, "test", 
        apply_aug=False
    )
    
    # Step 5: Print summary
    print_dataset_summary(output_path)
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║   ✅ DATASET PREPARATION COMPLETE!                                    ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    
    📁 Balanced dataset created at: {output_path}
    
    NEXT STEPS:
        1. Update config.py to use the new balanced dataset path
        2. Run the ML pipeline: python run_pipeline.py
    """)
    
    return output_path


def print_dataset_summary(dataset_path):
    """Print summary of the prepared dataset."""
    print("\n" + "=" * 70)
    print("📊 DATASET SUMMARY")
    print("=" * 70)
    
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            continue
        
        labels_dir = os.path.join(split_path, "labels")
        
        # Count samples per class
        class_counts = defaultdict(int)
        for label_file in glob.glob(os.path.join(labels_dir, "*.txt")):
            try:
                with open(label_file, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line:
                        class_id = int(first_line.split()[0])
                        class_counts[class_id] += 1
            except:
                continue
        
        total = sum(class_counts.values())
        print(f"\n   {split.upper()} SET ({total} samples):")
        for class_id in sorted(class_counts.keys()):
            count = class_counts[class_id]
            bar = "█" * (count // 50)
            print(f"      Class {class_id}: {count:5d} {bar}")


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_augmentations(image_path, output_path=None):
    """
    Visualize different augmentation techniques on a sample image.
    
    Parameters:
        image_path: Path to input image
        output_path: Path to save visualization (optional)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    # Original
    axes[0].imshow(image)
    axes[0].set_title("Original", fontweight='bold')
    axes[0].axis('off')
    
    # Horizontal flip
    axes[1].imshow(cv2.flip(image, 1))
    axes[1].set_title("Horizontal Flip")
    axes[1].axis('off')
    
    # Vertical flip
    axes[2].imshow(cv2.flip(image, 0))
    axes[2].set_title("Vertical Flip")
    axes[2].axis('off')
    
    # Rotation
    axes[3].imshow(cv2.cvtColor(random_rotation(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 25), cv2.COLOR_BGR2RGB))
    axes[3].set_title("Rotation (±25°)")
    axes[3].axis('off')
    
    # Brightness/Contrast
    axes[4].imshow(cv2.cvtColor(random_brightness_contrast(cv2.cvtColor(image, cv2.COLOR_RGB2BGR)), cv2.COLOR_BGR2RGB))
    axes[4].set_title("Brightness/Contrast")
    axes[4].axis('off')
    
    # Gaussian blur
    axes[5].imshow(cv2.cvtColor(cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), (5, 5), 0), cv2.COLOR_BGR2RGB))
    axes[5].set_title("Gaussian Blur")
    axes[5].axis('off')
    
    # Noise
    axes[6].imshow(cv2.cvtColor(random_noise(cv2.cvtColor(image, cv2.COLOR_RGB2BGR)), cv2.COLOR_BGR2RGB))
    axes[6].set_title("Noise Addition")
    axes[6].axis('off')
    
    # Color shift
    axes[7].imshow(cv2.cvtColor(random_color_shift(cv2.cvtColor(image, cv2.COLOR_RGB2BGR)), cv2.COLOR_BGR2RGB))
    axes[7].set_title("Color Shift")
    axes[7].axis('off')
    
    # Crop & Resize
    axes[8].imshow(cv2.cvtColor(random_crop_resize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.8), cv2.COLOR_BGR2RGB))
    axes[8].set_title("Crop & Resize")
    axes[8].axis('off')
    
    # Shear
    axes[9].imshow(cv2.cvtColor(random_shear(cv2.cvtColor(image, cv2.COLOR_RGB2BGR)), cv2.COLOR_BGR2RGB))
    axes[9].set_title("Shear Transform")
    axes[9].axis('off')
    
    # Light augmentation
    axes[10].imshow(cv2.cvtColor(apply_augmentation(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 'light'), cv2.COLOR_BGR2RGB))
    axes[10].set_title("Light Aug.", fontweight='bold', color='green')
    axes[10].axis('off')
    
    # Heavy augmentation
    axes[11].imshow(cv2.cvtColor(apply_augmentation(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 'heavy'), cv2.COLOR_BGR2RGB))
    axes[11].set_title("Heavy Aug.", fontweight='bold', color='red')
    axes[11].axis('off')
    
    plt.suptitle("Data Augmentation Techniques", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "step0_augmentation_demo.png")
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved augmentation demo to: {output_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run dataset preparation."""
    prepare_balanced_dataset()


if __name__ == "__main__":
    main()
