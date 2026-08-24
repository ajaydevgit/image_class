"""
OpenCV Image Processing and Enhancement Demonstration Script.

Task 18 Step 1: Preprocess images using OpenCV, apply enhancement techniques
(brightness/contrast adjustment, filtering, edge detection), and save side-by-side
comparisons for documentation.
"""

import os
import cv2
import numpy as np
from sklearn.datasets import load_digits
from src.image_processor import ImageProcessor
from src.utils import plot_side_by_side_comparison, plot_filters_comparison


def main():
    print("=" * 65)
    print("      OPENCV IMAGE PROCESSING & ENHANCEMENT DEMO (TASK 18)     ")
    print("=" * 65)

    # Load sample handwritten digit from sklearn dataset
    digits = load_digits()
    raw_sample = digits.images[0] # 8x8 raw image
    
    # Scale up sample for OpenCV image manipulation
    sample_uint8 = cv2.normalize(raw_sample, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    sample_28x28 = cv2.resize(sample_uint8, (28, 28), interpolation=cv2.INTER_CUBIC)

    print("\n1. Original Image Properties:")
    print(f"   - Shape: {sample_28x28.shape}")
    print(f"   - Data Type: {sample_28x28.dtype}")
    print(f"   - Min Pixel: {sample_28x28.min()}, Max Pixel: {sample_28x28.max()}")

    # --- 1. Grayscale & Resizing ---
    gray_img = ImageProcessor.to_grayscale(sample_28x28)
    resized_img = ImageProcessor.resize_image(gray_img, target_size=(28, 28))
    normalized_img = ImageProcessor.normalize_pixels(resized_img)

    print("\n2. Basic OpenCV Preprocessing Completed:")
    print("   - Converted to Grayscale (cv2.cvtColor)")
    print("   - Resized to 28x28 resolution (cv2.resize)")
    print("   - Normalized pixel range to [0.0, 1.0]")

    # --- 2. Brightness & Contrast Adjustment ---
    enhanced_img = ImageProcessor.adjust_brightness_contrast(sample_28x28, alpha=1.5, beta=40)
    print("\n3. Applied Brightness & Contrast Adjustment:")
    print("   - Formula: alpha (1.5) * pixel + beta (40) via cv2.convertScaleAbs")

    # Plot & save side-by-side brightness/contrast comparison
    plot_side_by_side_comparison(
        original_img=sample_28x28,
        enhanced_img=enhanced_img,
        original_title="Original Grayscale (28x28)",
        enhanced_title="Enhanced Contrast & Brightness",
        save_path="assets/brightness_contrast_comparison.png"
    )

    # --- 3. Image Filtering Techniques ---
    print("\n4. Applying OpenCV Image Filters:")
    gaussian = ImageProcessor.apply_gaussian_blur(sample_28x28, kernel_size=(5, 5), sigma_x=1.2)
    print("   - Gaussian Blur (cv2.GaussianBlur, kernel=5x5)")

    median = ImageProcessor.apply_median_blur(sample_28x28, kernel_size=3)
    print("   - Median Blur (cv2.medianBlur, kernel=3)")

    sharpened = ImageProcessor.apply_sharpening(sample_28x28)
    print("   - Sharpening Filter (cv2.filter2D)")

    canny = ImageProcessor.apply_canny_edge(sample_28x28, threshold1=50, threshold2=150)
    print("   - Canny Edge Detection (cv2.Canny, t1=50, t2=150)")

    # Plot & save multi-panel filter comparison
    filters_dict = {
        "Enhanced Brightness": enhanced_img,
        "Gaussian Blur": gaussian,
        "Median Blur": median,
        "Sharpened Filter": sharpened,
        "Canny Edge Map": canny
    }

    plot_filters_comparison(
        original_img=sample_28x28,
        filtered_images=filters_dict,
        save_path="assets/image_filters_comparison.png"
    )

    print("\n" + "=" * 65)
    print("   OpenCV Preprocessing Demo Finished. Assets saved in assets/  ")
    print("=" * 65)


if __name__ == "__main__":
    main()
