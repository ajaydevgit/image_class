"""
Main End-to-End Image Classification and OpenCV Processing Pipeline.

Task 18: Simple Image Classification & Image Processing with OpenCV
Demonstrates:
- Dataset loading and saving sample raw/processed images to disk
- OpenCV image preprocessing (grayscale, resize, contrast enhancement, normalization)
- Train-test splitting (80% train, 20% test)
- Model training with Support Vector Machine (SVM) and k-Nearest Neighbors (k-NN)
- Model evaluation (Accuracy, Precision, Recall, F1-score, Confusion Matrix)
- Exporting comparative figures to assets/ directory
"""

import sys
import numpy as np
from src.dataset import DatasetLoader
from src.image_processor import ImageProcessor
from src.model import ImageClassifier
from src.utils import (
    plot_side_by_side_comparison, 
    plot_filters_comparison, 
    plot_confusion_matrix, 
    plot_sample_predictions
)


def run_pipeline():
    print("=" * 70)
    print("  TASK 18: SIMPLE IMAGE CLASSIFICATION & IMAGE PROCESSING WITH OPENCV  ")
    print("=" * 70)

    # -------------------------------------------------------------------
    # STEP 1: Load Dataset & Save Sample Raw Images
    # -------------------------------------------------------------------
    print("\n[STEP 1] Loading Dataset & Preprocessing via OpenCV...")
    loader = DatasetLoader(raw_dir="data/raw", processed_dir="data/processed")
    data = loader.load_dataset(target_size=(28, 28), enhance=True, test_size=0.20, random_state=42)

    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    images_test = data["images_test"]
    target_names = data["target_names"]

    print(f"   - Total Training Samples : {X_train.shape[0]} (80% split)")
    print(f"   - Total Testing Samples  : {X_test.shape[0]} (20% split)")
    print(f"   - Feature Vector Dimension: {X_train.shape[1]} pixels (28x28 normalized image)")
    print(f"   - Number of Classes      : {len(target_names)} (Digits 0-9)")
    print(f"   - Raw images saved to    : data/raw/")
    print(f"   - Processed images saved to: data/processed/")

    # -------------------------------------------------------------------
    # STEP 2: Generate OpenCV Image Enhancement Visualizations
    # -------------------------------------------------------------------
    print("\n[STEP 2] Generating OpenCV Preprocessing & Filter Assets...")
    sample_raw = data["raw_images_sample"][0]
    # Scale up for visualization
    import cv2
    sample_scaled = cv2.resize((cv2.normalize(sample_raw, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8), (28, 28))

    enhanced_sample = ImageProcessor.adjust_brightness_contrast(sample_scaled, alpha=1.4, beta=30)
    gaussian_sample = ImageProcessor.apply_gaussian_blur(sample_scaled, kernel_size=(5, 5))
    median_sample = ImageProcessor.apply_median_blur(sample_scaled, kernel_size=3)
    sharpened_sample = ImageProcessor.apply_sharpening(sample_scaled)
    canny_sample = ImageProcessor.apply_canny_edge(sample_scaled, threshold1=50, threshold2=150)

    # Export Side-by-Side Brightness & Contrast
    plot_side_by_side_comparison(
        original_img=sample_scaled,
        enhanced_img=enhanced_sample,
        original_title="Original Image",
        enhanced_title="Enhanced Contrast & Brightness",
        save_path="assets/brightness_contrast_comparison.png"
    )

    # Export Multi-Filter Comparison
    filters_map = {
        "Enhanced Contrast": enhanced_sample,
        "Gaussian Blur": gaussian_sample,
        "Median Blur": median_sample,
        "Sharpened": sharpened_sample,
        "Canny Edges": canny_sample
    }
    plot_filters_comparison(
        original_img=sample_scaled,
        filtered_images=filters_map,
        save_path="assets/image_filters_comparison.png"
    )

    # -------------------------------------------------------------------
    # STEP 3: Train & Evaluate SVM Classifier
    # -------------------------------------------------------------------
    print("\n[STEP 3] Training Support Vector Machine (SVM) Model...")
    svm_classifier = ImageClassifier(model_type="svm", C=10.0, kernel="rbf")
    svm_classifier.fit(X_train, y_train)
    svm_results = svm_classifier.evaluate(X_test, y_test)

    print(f"   - SVM Accuracy : {svm_results['accuracy'] * 100:.2f}%")
    print(f"   - SVM Precision: {svm_results['precision'] * 100:.2f}%")
    print(f"   - SVM Recall   : {svm_results['recall'] * 100:.2f}%")
    print(f"   - SVM F1-Score : {svm_results['f1_score'] * 100:.2f}%")

    plot_confusion_matrix(
        cm=svm_results["confusion_matrix"],
        class_names=target_names,
        title="SVM Classifier - Confusion Matrix",
        save_path="assets/confusion_matrix_svm.png"
    )

    # -------------------------------------------------------------------
    # STEP 4: Train & Evaluate k-NN Classifier
    # -------------------------------------------------------------------
    print("\n[STEP 4] Training k-Nearest Neighbors (k-NN) Model...")
    knn_classifier = ImageClassifier(model_type="knn", n_neighbors=5, weights="distance")
    knn_classifier.fit(X_train, y_train)
    knn_results = knn_classifier.evaluate(X_test, y_test)

    print(f"   - k-NN Accuracy : {knn_results['accuracy'] * 100:.2f}%")
    print(f"   - k-NN Precision: {knn_results['precision'] * 100:.2f}%")
    print(f"   - k-NN Recall   : {knn_results['recall'] * 100:.2f}%")
    print(f"   - k-NN F1-Score : {knn_results['f1_score'] * 100:.2f}%")

    plot_confusion_matrix(
        cm=knn_results["confusion_matrix"],
        class_names=target_names,
        title="k-NN Classifier - Confusion Matrix",
        save_path="assets/confusion_matrix_knn.png"
    )

    # -------------------------------------------------------------------
    # STEP 5: Visualize Test Predictions
    # -------------------------------------------------------------------
    print("\n[STEP 5] Visualizing Sample Test Image Predictions...")
    plot_sample_predictions(
        test_images=images_test,
        y_true=y_test,
        y_pred=svm_results["y_pred"],
        num_samples=12,
        save_path="assets/sample_predictions.png"
    )

    # -------------------------------------------------------------------
    # STEP 6: Summary Comparison
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("                      MODEL PERFORMANCE SUMMARY                      ")
    print("=" * 70)
    print(f"{'Model Algorithm':<25} | {'Accuracy':<10} | {'Precision':<10} | {'F1-Score':<10}")
    print("-" * 70)
    print(f"{'Support Vector Machine (SVM)':<25} | {svm_results['accuracy']*100:6.2f}%    | {svm_results['precision']*100:6.2f}%    | {svm_results['f1_score']*100:6.2f}%")
    print(f"{'k-Nearest Neighbors (k-NN)':<25} | {knn_results['accuracy']*100:6.2f}%    | {knn_results['precision']*100:6.2f}%    | {knn_results['f1_score']*100:6.2f}%")
    print("=" * 70)
    print("\nPipeline executed successfully! All visual assets saved to assets/")


if __name__ == "__main__":
    run_pipeline()
