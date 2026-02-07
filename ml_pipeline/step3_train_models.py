"""
=============================================================================
STEP 3: TRAINING MACHINE LEARNING MODELS
=============================================================================

Trains and evaluates multiple ML classifiers on extracted features.

Key Functions:
    - train_all_models(): Train and compare all classifiers
    - get_best_model(): Get the best performing model
    - save_model(): Save trained model to disk
"""

import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

from config import OUTPUT_DIR, RANDOM_SEED

np.random.seed(RANDOM_SEED)


def get_classifiers():
    """Get dictionary of all classifiers to train."""
    return {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1),
        'SVM (RBF Kernel)': SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=RANDOM_SEED),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=RANDOM_SEED),
        'Neural Network (MLP)': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=RANDOM_SEED, early_stopping=True)
    }


def preprocess_features(X_train, X_test):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def train_single_model(clf, name, X_train, X_test, y_train, y_test):
    """Train a single classifier and evaluate."""
    print(f"\n   📈 Training {name}...")
    try:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        result = {
            'model': clf,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'predictions': y_pred,
            'report': classification_report(y_test, y_pred, output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        print(f"      ✅ Accuracy: {result['accuracy']:.4f}")
        return result
    except Exception as e:
        print(f"      ❌ Error: {str(e)}")
        return {'error': str(e)}


def train_all_models(X_train, X_test, y_train, y_test, feature_names=None):
    """Train all classifiers and compare performance."""
    print("\n🎯 TRAINING MACHINE LEARNING MODELS")
    print("=" * 60)
    
    X_train_scaled, X_test_scaled, scaler = preprocess_features(X_train, X_test)
    classifiers = get_classifiers()
    results = {}
    
    for name, clf in classifiers.items():
        results[name] = train_single_model(clf, name, X_train_scaled, X_test_scaled, y_train, y_test)
    
    print("\n" + "=" * 60)
    print("📊 RESULTS SUMMARY")
    print("-" * 60)
    for name, result in results.items():
        if 'accuracy' in result:
            print(f"   {name:24s} | Acc: {result['accuracy']:.4f} | F1: {result['f1']:.4f}")
    
    return results, scaler


def get_best_model(results):
    """Get the best performing model based on accuracy."""
    best_name, best_accuracy = None, 0
    for name, result in results.items():
        if 'accuracy' in result and result['accuracy'] > best_accuracy:
            best_accuracy, best_name = result['accuracy'], name
    
    if best_name:
        print(f"\n🏆 BEST MODEL: {best_name} (Accuracy: {best_accuracy:.4f})")
        return results[best_name]['model'], best_name, best_accuracy
    return None, None, 0


def get_feature_importance(results, feature_names):
    """Get feature importance from Random Forest."""
    if 'Random Forest' in results and 'model' in results['Random Forest']:
        importances = results['Random Forest']['model'].feature_importances_
        importance_list = sorted(zip(feature_names, importances), key=lambda x: -x[1])
        return importance_list
    return None


def save_model(model, scaler, model_name="unknown", filename="best_model.pkl"):
    """
    Save trained model and scaler to disk.
    
    Saves as .pkl (pickle) format - standard for scikit-learn models.
    Note: .pth is PyTorch format, .pkl is sklearn format.
    """
    save_path = os.path.join(OUTPUT_DIR, filename)
    model_data = {
        'model': model,
        'scaler': scaler,
        'model_name': model_name,
        'library': 'scikit-learn'
    }
    joblib.dump(model_data, save_path)
    print(f"\n💾 BEST MODEL SAVED:")
    print(f"   File: {save_path}")
    print(f"   Model Type: {model_name}")
    print(f"   Format: .pkl (scikit-learn / joblib)")
    return save_path


def load_model(filename="best_model.pkl"):
    """Load trained model and scaler from disk."""
    load_path = os.path.join(OUTPUT_DIR, filename)
    model_data = joblib.load(load_path)
    return model_data['model'], model_data['scaler']


if __name__ == "__main__":
    print("Run this module via run_pipeline.py")
