import cv2
import numpy as np
import os

# ============================================================
# TWEAKABLE PARAMETERS
# ============================================================
# Circle detection parameters
HOUGH_PARAM1 = 50       # Higher threshold for the Canny edge detector
HOUGH_PARAM2 = 30       # Accumulator threshold for circle centers (lower = more circles)
CIRCLE_PAD = -5         # Inward padding to avoid capturing background pixels at the disc edge

# Contrast Enhancement parameters
BIFILTER_D = 9          # Diameter of each pixel neighborhood for smoothing
BIFILTER_SIGMA = 75     # Filter sigma for color and space (higher = more blur/grain removal)
CLAHE_CLIP = 4.0        # Threshold for contrast limiting (higher = more aggressive contrast)
CLAHE_GRID = (8, 8)     # Size of the grid for histogram equalization

# Intensity Clipping parameters
LOWER_PERCENTILE = 5    # Percentage of darkest pixels to completely flatten to black
UPPER_PERCENTILE = 95   # Percentage of brightest pixels to completely push to white
# ============================================================

# Ensure output directory exists
output_dir = "preprocessing_outputs"
os.makedirs(output_dir, exist_ok=True)

# 1. Load Image
img = cv2.imread("image_6a7bfc.jpg")
if img is None:
    raise FileNotFoundError("Could not find the input image file.")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height, width = img_gray.shape[:2]

# 2. Smooth slightly for clean circle detection
blur = cv2.GaussianBlur(img_gray, (9, 9), 2)

# 3. Detect the circular disc
circles = cv2.HoughCircles(
    blur,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=img.shape[0] // 2,
    param1=HOUGH_PARAM1,
    param2=HOUGH_PARAM2,
    minRadius=0,
    maxRadius=0
)

if circles is not None:
    circles = np.uint16(np.around(circles))
    cx, cy, r = circles[0][0][0], circles[0][0][1], circles[0][0][2]
    
    # Create mask slightly smaller than the detected circle to eliminate edge glare
    mask = np.zeros_like(img_gray)
    cv2.circle(mask, (cx, cy), r + CIRCLE_PAD, 255, -1)
else:
    print("Warning: No circle detected. Defaulting to full image mask.")
    mask = np.ones_like(img_gray) * 255

mask_bool = mask > 0

# Save masked original grayscale image
img_masked = np.zeros_like(img_gray)
img_masked[mask_bool] = img_gray[mask_bool]
cv2.imwrite(os.path.join(output_dir, "1_masked_original.jpg"), img_masked)

# 4. Bilateral Filter (Removes metal grain noise while keeping hard groove edges sharp)
filtered = np.zeros_like(img_gray)
filtered_zone = cv2.bilateralFilter(img_gray, d=BIFILTER_D, sigmaColor=BIFILTER_SIGMA, sigmaSpace=BIFILTER_SIGMA)
filtered[mask_bool] = filtered_zone[mask_bool]
cv2.imwrite(os.path.join(output_dir, "2_grain_removed.jpg"), filtered)

# 5. Local Adaptive Contrast Enhancement (CLAHE)
clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
clahe_out = np.zeros_like(img_gray)
clahe_out[mask_bool] = clahe.apply(filtered[mask_bool])
cv2.imwrite(os.path.join(output_dir, "3_clahe_enhanced.jpg"), clahe_out)

# 6. Aggressive Intensity Clipping (Stretches the remaining dynamic range to the absolute extremes)
disc_pixels = clahe_out[mask_bool]
low_val = np.percentile(disc_pixels, LOWER_PERCENTILE)
high_val = np.percentile(disc_pixels, UPPER_PERCENTILE)

# Rescale values between low_val and high_val to [0, 255]
final_preprocessed = np.zeros_like(img_gray)
clipped = np.clip(clahe_out, low_val, high_val)
rescaled = ((clipped - low_val) / (high_val - low_val) * 255).astype(np.uint8)
final_preprocessed[mask_bool] = rescaled[mask_bool]

# Save final result
cv2.imwrite(os.path.join(output_dir, "4_final_high_contrast.jpg"), final_preprocessed)
print(f"Preprocessing complete. Check the '{output_dir}' folder for the intermediate steps.")
