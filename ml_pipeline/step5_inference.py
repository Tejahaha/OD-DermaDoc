"""
=============================================================================
STEP 5: INFERENCE ON NEW IMAGES
=============================================================================

Use the trained model to classify new, unseen images.

Key Functions:
    - predict_single_image(): Classify a single image
    - predict_batch(): Classify multiple images
    - visualize_prediction(): Show prediction with confidence
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import OUTPUT_DIR, IMAGE_SIZE
from step2_extract_features import extract_all_features
from step3_train_models import load_model


def predict_single_image(image_path, model=None, scaler=None):
    """
    Classify a single image using the trained model.
    
    Parameters:
        image_path (str): Path to the image file
        model: Trained classifier (loads from disk if None)
        scaler: Fitted scaler (loads from disk if None)
    
    Returns:
        prediction (int): Predicted class ID
        confidence (dict or None): Class probabilities if available
    """
    # Load model if not provided
    if model is None or scaler is None:
        model, scaler = load_model("best_model.pkl")
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Extract features
    features, _ = extract_all_features(image)
    features = features.reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    
    # Get confidence if available
    confidence = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(features_scaled)[0]
        confidence = {i: p for i, p in enumerate(proba)}
    
    return prediction, confidence


def predict_batch(image_paths, model=None, scaler=None):
    """
    Classify multiple images.
    
    Parameters:
        image_paths (list): List of image file paths
        model: Trained classifier
        scaler: Fitted scaler
    
    Returns:
        predictions (list): List of predicted class IDs
        confidences (list): List of confidence dictionaries
    """
    if model is None or scaler is None:
        model, scaler = load_model()
    
    predictions = []
    confidences = []
    
    print(f"\n🔮 Predicting {len(image_paths)} images...")
    
    for i, path in enumerate(image_paths):
        try:
            pred, conf = predict_single_image(path, model, scaler)
            predictions.append(pred)
            confidences.append(conf)
            
            if (i + 1) % 10 == 0:
                print(f"   Processed: {i + 1}/{len(image_paths)}")
        except Exception as e:
            print(f"   ⚠️ Error with {path}: {e}")
            predictions.append(None)
            confidences.append(None)
    
    return predictions, confidences


def visualize_prediction(image_path, prediction, confidence=None, true_label=None, save_path=None):
    """
    Visualize prediction result for a single image.
    
    Parameters:
        image_path (str): Path to the image
        prediction (int): Predicted class
        confidence (dict): Class probabilities
        true_label (int): Actual class (if known)
        save_path (str): Where to save visualization
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Image with prediction
    axes[0].imshow(image)
    title = f"Predicted: Class {prediction}"
    if true_label is not None:
        title += f"\nActual: Class {true_label}"
        color = 'green' if prediction == true_label else 'red'
        axes[0].set_title(title, fontsize=12, fontweight='bold', color=color)
    else:
        axes[0].set_title(title, fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Confidence bar chart
    if confidence:
        classes = sorted(confidence.keys())
        probs = [confidence[c] for c in classes]
        colors = ['steelblue' if c != prediction else 'coral' for c in classes]
        
        axes[1].barh([f"Class {c}" for c in classes], probs, color=colors)
        axes[1].set_xlabel('Probability')
        axes[1].set_title('Class Probabilities', fontweight='bold')
        axes[1].set_xlim([0, 1])
    else:
        axes[1].text(0.5, 0.5, 'Confidence not available\n(model type does not support probabilities)',
                     ha='center', va='center', fontsize=12)
        axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "step5_prediction.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved prediction visualization: {save_path}")
    return save_path


def demo_inference(test_images, test_labels, model, scaler, num_samples=5):
    """
    Demonstrate inference on sample test images.
    
    Parameters:
        test_images (list): List of image paths
        test_labels (list): True labels
        model: Trained model
        scaler: Fitted scaler
        num_samples (int): Number of samples to demonstrate
    """
    print("\n🔮 INFERENCE DEMONSTRATION")
    print("=" * 50)
    
    indices = np.random.choice(len(test_images), min(num_samples, len(test_images)), replace=False)
    
    correct = 0
    for i, idx in enumerate(indices):
        image_path = test_images[idx]
        true_label = test_labels[idx]
        
        pred, conf = predict_single_image(image_path, model, scaler)
        
        status = "✅" if pred == true_label else "❌"
        if pred == true_label:
            correct += 1
        
        print(f"\n   Sample {i+1}:")
        print(f"      Image: {os.path.basename(image_path)}")
        print(f"      True:  Class {true_label}")
        print(f"      Pred:  Class {pred} {status}")
        
        if conf:
            top_conf = max(conf.values())
            print(f"      Confidence: {top_conf:.2%}")
        
        # Save visualization for first sample
        if i == 0:
            visualize_prediction(image_path, pred, conf, true_label)
    
    print(f"\n   Demo accuracy: {correct}/{len(indices)} ({100*correct/len(indices):.1f}%)")


if __name__ == "__main__":
    print("Run this module via run_pipeline.py")
