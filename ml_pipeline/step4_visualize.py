"""
=============================================================================
STEP 4: VISUALIZATION AND RESULTS ANALYSIS
=============================================================================

Creates visualizations to understand model performance and feature importance.

Key Functions:
    - plot_model_comparison(): Compare all models' performance
    - plot_confusion_matrix(): Visualize confusion matrix
    - plot_feature_importance(): Show top important features
    - generate_full_report(): Create complete visualization report
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from config import OUTPUT_DIR


def plot_model_comparison(results, save_path=None):
    """Create bar chart comparing model accuracies."""
    model_names = []
    accuracies = []
    f1_scores = []
    
    for name, result in results.items():
        if 'accuracy' in result:
            model_names.append(name)
            accuracies.append(result['accuracy'])
            f1_scores.append(result['f1'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue')
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1 Score', color='coral')
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "step4_model_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {save_path}")
    return save_path


def plot_confusion_matrix(confusion_mat, class_labels=None, model_name="Model", save_path=None):
    """Visualize confusion matrix as heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if class_labels is None:
        class_labels = [f"Class {i}" for i in range(len(confusion_mat))]
    
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "step4_confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {save_path}")
    return save_path


def plot_feature_importance(importance_list, top_n=15, save_path=None):
    """Plot top N most important features."""
    top_features = importance_list[:top_n]
    names = [f[0] for f in top_features]
    values = [f[1] for f in top_features]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
    bars = ax.barh(range(len(names)), values, color=colors)
    
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "step4_feature_importance.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {save_path}")
    return save_path


def plot_class_predictions(y_true, y_pred, save_path=None):
    """Plot prediction distribution vs true labels."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # True distribution
    classes, counts = np.unique(y_true, return_counts=True)
    axes[0].bar([f"Class {c}" for c in classes], counts, color='steelblue')
    axes[0].set_title('True Label Distribution', fontweight='bold')
    axes[0].set_ylabel('Count')
    
    # Predicted distribution
    pred_classes, pred_counts = np.unique(y_pred, return_counts=True)
    axes[1].bar([f"Class {c}" for c in pred_classes], pred_counts, color='coral')
    axes[1].set_title('Predicted Label Distribution', fontweight='bold')
    axes[1].set_ylabel('Count')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "step4_predictions.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {save_path}")
    return save_path


def generate_full_report(results, best_name, feature_names=None, y_test=None):
    """Generate all visualizations."""
    print("\n📊 GENERATING VISUALIZATIONS")
    print("-" * 50)
    
    # Model comparison
    plot_model_comparison(results)
    
    # Confusion matrix for best model
    if best_name and best_name in results:
        cm = results[best_name]['confusion_matrix']
        plot_confusion_matrix(cm, model_name=best_name)
        
        # Prediction distribution
        if y_test is not None:
            y_pred = results[best_name]['predictions']
            plot_class_predictions(y_test, y_pred)
    
    # Feature importance (from Random Forest)
    if feature_names and 'Random Forest' in results:
        from step3_train_models import get_feature_importance
        importance_list = get_feature_importance(results, feature_names)
        if importance_list:
            plot_feature_importance(importance_list)
    
    print(f"\n   All visualizations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    print("Run this module via run_pipeline.py")
