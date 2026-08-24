"""
Image Processing and Enhancement Module using OpenCV.

This module provides standard computer vision preprocessing operations including:
- Grayscale Conversion
- Image Resizing
- Pixel Value Normalization
- Brightness & Contrast Adjustments
- Image Filtering (Gaussian Blur, Median Blur, Sharpening, Sobel/Canny Edge Detection)
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Union


class ImageProcessor:
    """OpenCV Image Preprocessing and Enhancement Pipeline."""

    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """
        Convert RGB/BGR image to single-channel Grayscale.
        
        Args:
            image (np.ndarray): Input image array (2D or 3D).
            
        Returns:
            np.ndarray: Grayscale image (2D).
        """
        if len(image.shape) == 2:
            return image.copy()
        elif image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return image

    @staticmethod
    def resize_image(
        image: np.ndarray, 
        target_size: Tuple[int, int] = (28, 28), 
        interpolation: int = cv2.INTER_AREA
    ) -> np.ndarray:
        """
        Resize image to standard target dimensions.
        
        Args:
            image (np.ndarray): Input image.
            target_size (Tuple[int, int]): (width, height) target resolution.
            interpolation (int): OpenCV interpolation flag.
            
        Returns:
            np.ndarray: Resized image.
        """
        return cv2.resize(image, target_size, interpolation=interpolation)

    @staticmethod
    def normalize_pixels(image: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values to range [0.0, 1.0].
        
        Args:
            image (np.ndarray): Input image with values in range [0, 255].
            
        Returns:
            np.ndarray: Float32 normalized image.
        """
        image_float = image.astype(np.float32)
        if image_float.max() > 1.0:
            return image_float / 255.0
        return image_float

    @staticmethod
    def adjust_brightness_contrast(
        image: np.ndarray, 
        alpha: float = 1.2, 
        beta: int = 30
    ) -> np.ndarray:
        """
        Adjust Brightness and Contrast of an image using OpenCV convertScaleAbs.
        Formula: output_pixel = alpha * input_pixel + beta
        
        Args:
            image (np.ndarray): Input uint8 image.
            alpha (float): Contrast control (>0, 1.0 = no change).
            beta (int): Brightness control (integer added to pixels).
            
        Returns:
            np.ndarray: Enhanced image.
        """
        # Ensure image is in uint8 format for convertScaleAbs
        if image.dtype != np.uint8:
            img_uint8 = (image * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img_uint8 = image
        return cv2.convertScaleAbs(img_uint8, alpha=alpha, beta=beta)

    @staticmethod
    def apply_gaussian_blur(
        image: np.ndarray, 
        kernel_size: Tuple[int, int] = (5, 5), 
        sigma_x: float = 1.0
    ) -> np.ndarray:
        """
        Apply Gaussian Blurring for noise reduction.
        
        Args:
            image (np.ndarray): Input image.
            kernel_size (Tuple[int, int]): Gaussian kernel window size (must be odd integers).
            sigma_x (float): Gaussian kernel standard deviation in X direction.
            
        Returns:
            np.ndarray: Smoothed image.
        """
        return cv2.GaussianBlur(image, kernel_size, sigmaX=sigma_x)

    @staticmethod
    def apply_median_blur(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Apply Median Filter for salt-and-pepper noise removal.
        
        Args:
            image (np.ndarray): Input image.
            kernel_size (int): Kernel size (must be odd integer > 1).
            
        Returns:
            np.ndarray: Filtered image.
        """
        return cv2.medianBlur(image, kernel_size)

    @staticmethod
    def apply_sharpening(image: np.ndarray) -> np.ndarray:
        """
        Apply a sharpening filter using custom 2D convolution kernel.
        
        Args:
            image (np.ndarray): Input image.
            
        Returns:
            np.ndarray: Sharpened image.
        """
        sharpen_kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        return cv2.filter2D(image, -1, sharpen_kernel)

    @staticmethod
    def apply_canny_edge(
        image: np.ndarray, 
        threshold1: int = 100, 
        threshold2: int = 200
    ) -> np.ndarray:
        """
        Apply Canny Edge Detection algorithm.
        
        Args:
            image (np.ndarray): Input image.
            threshold1 (int): First threshold for hysteresis procedure.
            threshold2 (int): Second threshold for hysteresis procedure.
            
        Returns:
            np.ndarray: Binary edge map.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        if gray.dtype != np.uint8:
            gray = (gray * 255.0).clip(0, 255).astype(np.uint8)
        return cv2.Canny(gray, threshold1, threshold2)

    @classmethod
    def preprocess_pipeline(
        cls, 
        image: np.ndarray, 
        target_size: Tuple[int, int] = (28, 28),
        enhance: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Complete OpenCV Image Preprocessing Pipeline.
        
        1. Grayscale Conversion
        2. Optional Enhancement (Contrast/Brightness)
        3. Resizing
        4. Normalization
        
        Args:
            image (np.ndarray): Raw input image array.
            target_size (Tuple[int, int]): Dimensions to resize image to.
            enhance (bool): Whether to apply contrast & brightness enhancement.
            normalize (bool): Whether to normalize pixel values to [0, 1].
            
        Returns:
            np.ndarray: Fully preprocessed image.
        """
        # Step 1: Grayscale
        gray = cls.to_grayscale(image)

        # Step 2: Optional Enhancement
        if enhance:
            processed = cls.adjust_brightness_contrast(gray, alpha=1.2, beta=15)
        else:
            processed = gray

        # Step 3: Resize
        resized = cls.resize_image(processed, target_size=target_size)

        # Step 4: Normalize
        if normalize:
            final_img = cls.normalize_pixels(resized)
        else:
            final_img = resized

        return final_img

    @classmethod
    def process_batch(
        cls, 
        images: np.ndarray, 
        target_size: Tuple[int, int] = (28, 28),
        enhance: bool = False
    ) -> np.ndarray:
        """
        Preprocess a collection of images in batch.
        
        Args:
            images (np.ndarray): Batch array of images (N, H, W) or (N, H, W, C).
            target_size (Tuple[int, int]): Dimensions for resizing.
            enhance (bool): Whether to apply enhancement.
            
        Returns:
            np.ndarray: Preprocessed images array of shape (N, H, W).
        """
        processed_list = []
        for img in images:
            p_img = cls.preprocess_pipeline(img, target_size=target_size, enhance=enhance, normalize=True)
            processed_list.append(p_img)
        return np.array(processed_list)
