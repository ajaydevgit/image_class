"""
Dataset Loader and Manager Module.

Loads labeled standard image datasets, saves sample images to disk for OpenCV processing,
performs train-test splitting, and formats image matrices into 1D feature vectors.
"""

import os
import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from src.image_processor import ImageProcessor


class DatasetLoader:
    """Manages dataset collection, preprocessing, disk saving, and feature splitting."""

    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        """
        Initialize DatasetLoader.
        
        Args:
            raw_dir (str): Directory path to save raw sample images.
            processed_dir (str): Directory path to save processed sample images.
        """
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def load_dataset(
        self, 
        target_size: Tuple[int, int] = (28, 28),
        enhance: bool = True,
        test_size: float = 0.20,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Load standard Handwritten Digits dataset, preprocess images via OpenCV,
        save sample image files to disk, and split into train/test sets.
        
        Args:
            target_size (Tuple[int, int]): Target image dimensions for OpenCV resizing.
            enhance (bool): Whether to apply OpenCV contrast/brightness enhancement.
            test_size (float): Proportion of dataset for test set (0.20 = 20%).
            random_state (int): Random seed for reproducible splitting.
            
        Returns:
            Dict[str, Any]: Dictionary containing train/test split data, raw/processed images, and metadata.
        """
        # Load sklearn Digits dataset (1797 samples, 8x8 grayscale images, 10 classes)
        digits = load_digits()
        raw_images = digits.images # (1797, 8, 8)
        labels = digits.target       # (1797,)

        # Save a sample of raw images to disk as PNG files for OpenCV demonstrations
        self._save_sample_images(raw_images, labels, num_samples=10)

        # Apply OpenCV batch preprocessing (Grayscale, Enhance, Resize, Normalize)
        processed_images = []
        for img in raw_images:
            # Upscale uint8 image for OpenCV operations
            img_uint8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            p_img = ImageProcessor.preprocess_pipeline(
                img_uint8, 
                target_size=target_size, 
                enhance=enhance, 
                normalize=True
            )
            processed_images.append(p_img)

        processed_images = np.array(processed_images) # (1797, 28, 28)

        # Save preprocessed samples to disk
        self._save_processed_samples(processed_images, labels, num_samples=10)

        # Flatten 2D image matrices into 1D feature vectors for ML models
        n_samples = len(processed_images)
        X_features = processed_images.reshape(n_samples, -1) # (1797, 28*28)

        # Stratified 80% Train / 20% Test Split
        X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
            X_features, 
            labels, 
            processed_images,
            test_size=test_size, 
            random_state=random_state, 
            stratify=labels
        )

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "images_train": img_train,
            "images_test": img_test,
            "target_names": [str(i) for i in range(10)],
            "raw_images_sample": raw_images[:10],
            "labels_sample": labels[:10]
        }

    def _save_sample_images(self, images: np.ndarray, labels: np.ndarray, num_samples: int = 10):
        """Save sample raw images to data/raw directory."""
        for i in range(min(num_samples, len(images))):
            img_uint8 = cv2.normalize(images[i], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            # Upscale for better viewing
            img_scaled = cv2.resize(img_uint8, (128, 128), interpolation=cv2.INTER_NEAREST)
            path = os.path.join(self.raw_dir, f"sample_{i}_label_{labels[i]}.png")
            cv2.imwrite(path, img_scaled)

    def _save_processed_samples(self, images: np.ndarray, labels: np.ndarray, num_samples: int = 10):
        """Save sample preprocessed images to data/processed directory."""
        for i in range(min(num_samples, len(images))):
            img_uint8 = (images[i] * 255.0).clip(0, 255).astype(np.uint8)
            img_scaled = cv2.resize(img_uint8, (128, 128), interpolation=cv2.INTER_NEAREST)
            path = os.path.join(self.processed_dir, f"processed_{i}_label_{labels[i]}.png")
            cv2.imwrite(path, img_scaled)
