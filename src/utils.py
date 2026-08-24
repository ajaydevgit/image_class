"""
Visualization and Metric Reporting Utilities.

Provides functions to plot and export:
- Side-by-side OpenCV image processing comparisons
- Image filter multi-panel grids
- Confusion matrix heatmaps
- Sample test image predictions with actual vs predicted labels
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional


def plot_side_by_side_comparison(
    original_img: np.ndarray,
    enhanced_img: np.ndarray,
    original_title: str = "Original Image",
    enhanced_title: str = "Enhanced (Brightness & Contrast)",
    save_path: Optional[str] = "assets/brightness_contrast_comparison.png"
):
    """
    Plot and save side-by-side comparison of original vs OpenCV preprocessed image.
    
    Args:
        original_img (np.ndarray): Original image array.
        enhanced_img (np.ndarray): OpenCV enhanced image array.
        original_title (str): Title for original subplot.
        enhanced_title (str): Title for enhanced subplot.
        save_path (Optional[str]): Output filepath to save figure.
    """
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot Original
    axes[0].imshow(original_img, cmap='gray' if len(original_img.shape) == 2 else None)
    axes[0].set_title(original_title, fontsize=12, fontweight='bold', pad=10)
    axes[0].axis('off')

    # Plot Enhanced
    axes[1].imshow(enhanced_img, cmap='gray' if len(enhanced_img.shape) == 2 else None)
    axes[1].set_title(enhanced_title, fontsize=12, fontweight='bold', pad=10)
    axes[1].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved side-by-side comparison figure to: {save_path}")
    plt.close(fig)


def plot_filters_comparison(
    original_img: np.ndarray,
    filtered_images: Dict[str, np.ndarray],
    save_path: Optional[str] = "assets/image_filters_comparison.png"
):
    """
    Plot grid of original image alongside various OpenCV filter outputs
    (Gaussian Blur, Median Blur, Sharpened, Canny Edges).
    
    Args:
        original_img (np.ndarray): Input original image.
        filtered_images (Dict[str, np.ndarray]): Dict mapping filter names to images.
        save_path (Optional[str]): Output filepath.
    """
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    total_images = len(filtered_images) + 1
    cols = 3
    rows = (total_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten()

    # Original
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title("Original Grayscale", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    # Filtered outputs
    for i, (title, img) in enumerate(filtered_images.items(), start=1):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(title, fontsize=11, fontweight='bold')
        axes[i].axis('off')

    # Turn off unused subplots
    for j in range(total_images, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved filters comparison figure to: {save_path}")
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = "assets/confusion_matrix.png"
):
    """
    Plot and save Matplotlib confusion matrix heatmap with text annotations.
    
    Args:
        cm (np.ndarray): Confusion matrix array.
        class_names (List[str]): Class label names.
        title (str): Plot title.
        save_path (Optional[str]): Output file path.
    """
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Set tick labels
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names, 
        yticklabels=class_names,
        title=title,
        ylabel='True Label',
        xlabel='Predicted Label'
    )
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11, fontweight='bold')

    # Annotate numbers in heatmap cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10, fontweight='bold'
            )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix plot to: {save_path}")
    plt.close(fig)


def plot_sample_predictions(
    test_images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_samples: int = 12,
    save_path: Optional[str] = "assets/sample_predictions.png"
):
    """
    Plot a grid of sample test images with their predicted and actual labels.
    Color codes title green for correct classification and red for misclassification.
    
    Args:
        test_images (np.ndarray): 2D/3D test image arrays.
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        num_samples (int): Number of sample images to display.
        save_path (Optional[str]): Output file path.
    """
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cols = 4
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows))
    axes = axes.flatten()

    for i in range(min(num_samples, len(test_images))):
        img = test_images[i]
        true_lbl = y_true[i]
        pred_lbl = y_pred[i]

        axes[i].imshow(img, cmap='gray')
        is_correct = (true_lbl == pred_lbl)
        color = 'green' if is_correct else 'red'
        status = '✓' if is_correct else '✗'

        title_str = f"Pred: {pred_lbl} | True: {true_lbl} {status}"
        axes[i].set_title(title_str, color=color, fontsize=11, fontweight='bold')
        axes[i].axis('off')

    # Turn off unused subplots
    for j in range(num_samples, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved sample predictions plot to: {save_path}")
    plt.close(fig)
