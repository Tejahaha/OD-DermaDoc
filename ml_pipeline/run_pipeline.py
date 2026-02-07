"""
=============================================================================
RUN COMPLETE ML PIPELINE
=============================================================================

This is the main entry point for running the complete machine learning
pipeline for skin lesion classification.

PIPELINE STEPS:
    1. Load images and labels from dataset
    2. Extract features (color, texture, shape, statistical)
    3. Train and evaluate ML models
    4. Visualize results
    5. Demonstrate inference on test images

Usage:
    python run_pipeline.py

Or import and run:
    from run_pipeline import run_full_pipeline
    run_full_pipeline()
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TRAIN_PATH, VALID_PATH, TEST_PATH, OUTPUT_DIR,
    MAX_TRAIN_SAMPLES, MAX_VALID_SAMPLES
)


def print_banner():
    """Print the pipeline ba
    zxsznner."""
    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║   🔬 TRADITIONAL ML PIPELINE FOR SKIN LESION CLASSIFICATION           ║
    ║                                                                       ║
    ║   A modular, step-by-step approach to image classification            ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    
    Pipeline Overview:
    
    📋 STEP 1: Load Data        → Load images with class labels
    🔧 STEP 2: Extract Features → Color, texture, shape, statistics
    🎯 STEP 3: Train Models     → Random Forest, SVM, KNN, etc.
    📊 STEP 4: Visualize        → Charts and confusion matrices  
    🔮 STEP 5: Inference        → Predict on new images
    
    """)


def run_full_pipeline():
    """
    Run the complete ML pipeline from start to finish.

    Returns:
        results: Training results dictionary
        best_model: Best performing model
        scaler: Fitted feature scaler
    """
    print_banner()

    # =========================================================================
    # STEP 1: LOAD DATA
    # =========================================================================
    print("\n" + "=" * 70)
    print("📋 STEP 1: LOADING IMAGES AND LABELS")
    print("=" * 70)

    from step1_load_data import (
        load_images_with_labels,
        print_class_distribution,
        visualize_sample_images,
        visualize_class_distribution
    )

    # Load training data
    print("\n   Loading training data...")
    train_images, train_labels, train_paths = load_images_with_labels(
        TRAIN_PATH, MAX_TRAIN_SAMPLES
    )

    # Load validation data
    print("\n   Loading validation data...")
    val_images, val_labels, val_paths = load_images_with_labels(
        VALID_PATH, MAX_VALID_SAMPLES
    )

    # Print class distribution
    print_class_distribution(train_labels, "Training Set")

    # Visualize samples
    visualize_sample_images(train_images, train_labels)
    visualize_class_distribution(train_labels)

    print("\n   ✅ STEP 1 COMPLETE!")
    print(f"      Training samples: {len(train_images)}")
    print(f"      Validation samples: {len(val_images)}")

    # =========================================================================
    # STEP 2: EXTRACT FEATURES
    # =========================================================================
    print("\n" + "=" * 70)
    print("🔧 STEP 2: EXTRACTING FEATURES")
    print("=" * 70)
    print("""
    Extracting:
    • Color features (RGB/HSV statistics, histograms)
    • Texture features (GLCM: contrast, homogeneity, etc.)
    • Shape features (edges, circularity, area)
    • Statistical features (entropy, skewness, kurtosis)
    """)

    from step2_extract_features import (
        extract_features_from_dataset,
        visualize_feature_extraction_example,
        print_feature_summary
    )

    # Visualize feature extraction on sample image
    if train_images:
        visualize_feature_extraction_example(train_images[0])

    # Extract features from training set
    print("\n   Processing training set...")
    X_train, y_train, feature_names = extract_features_from_dataset(
        train_images, train_labels
    )

    # Extract features from validation set
    print("\n   Processing validation set...")
    X_val, y_val, _ = extract_features_from_dataset(val_images, val_labels)

    # Print feature summary
    print_feature_summary(feature_names)

    print("\n   ✅ STEP 2 COMPLETE!")
    print(f"      Feature vector size: {len(feature_names)}")
    print(f"      Training matrix: {X_train.shape}")

    # =========================================================================
    # STEP 3: TRAIN MODELS
    # =========================================================================
    print("\n" + "=" * 70)
    print("🎯 STEP 3: TRAINING MACHINE LEARNING MODELS")
    print("=" * 70)

    from step3_train_models import (
        train_all_models,
        get_best_model,
        get_feature_importance,
        save_model
    )

    # Train all models
    results, scaler = train_all_models(X_train, X_val, y_train, y_val, feature_names)

    # Get best model
    best_model, best_name, best_accuracy = get_best_model(results)

    # Get feature importance
    importance_list = get_feature_importance(results, feature_names)
    if importance_list:
        print("\n   📊 TOP 10 FEATURES:")
        for i, (name, imp) in enumerate(importance_list[:10]):
            print(f"      {i+1:2d}. {name:25s} {imp:.4f}")

    # Save best model
    if best_model:
        save_model(best_model, scaler, model_name=best_name)

    print("\n   ✅ STEP 3 COMPLETE!")

    # =========================================================================
    # STEP 4: VISUALIZE RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 STEP 4: GENERATING VISUALIZATIONS")
    print("=" * 70)

    from step4_visualize import generate_full_report

    generate_full_report(results, best_name, feature_names, y_val)

    print("\n   ✅ STEP 4 COMPLETE!")

    # =========================================================================
    # STEP 5: INFERENCE DEMO
    # =========================================================================
    print("\n" + "=" * 70)
    print("🔮 STEP 5: INFERENCE DEMONSTRATION")
    print("=" * 70)

    from step5_inference import demo_inference

    if best_model:
        demo_inference(val_paths, val_labels, best_model, scaler, num_samples=5)

    print("\n   ✅ STEP 5 COMPLETE!")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║   🎉 PIPELINE COMPLETE!                                               ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)

    print(f"   📁 Results saved to: {OUTPUT_DIR}")
    print(f"   🏆 Best Model: {best_name} (Accuracy: {best_accuracy:.4f})")
    print("""
    📄 OUTPUT FILES:
       • step1_sample_images.png      - Sample images visualization
       • step1_class_distribution.png - Class distribution chart
       • step2_feature_visualization.png - Feature extraction demo
       • step4_model_comparison.png   - Model accuracy comparison
       • step4_confusion_matrix.png   - Confusion matrix heatmap
       • step4_feature_importance.png - Top features bar chart
       • step5_prediction.png         - Sample prediction visualization
       • best_model.pkl               - Saved model for inference
    
    🚀 NEXT STEPS:
       1. Increase training samples (remove MAX_TRAIN_SAMPLES limit)
       2. Try hyperparameter tuning with GridSearchCV
       3. Add more domain-specific features
       4. Experiment with ensemble methods
    """)

    return results, best_model, scaler


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_full_pipeline()
