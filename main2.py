import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def analyze_disc_degradation(image_path, num_color_groups=3):
    # 1. Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image. Check the file path.")
        return

    # Convert BGR (OpenCV default) to RGB for accurate colour processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Create Masks
    # Mask out the background (Assuming a light/white background, adjust if transparent)
    _, bg_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Mask out the black slots (Very dark pixels)
    _, slots_mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    
    # The valid disc area is where it is NOT background and NOT a black slot
    valid_mask = cv2.bitwise_and(bg_mask, slots_mask)
    
    # Convert mask to boolean for easier array indexing
    valid_mask_bool = valid_mask > 0

    # 3. Extract valid pixels for clustering
    valid_pixels = image_rgb[valid_mask_bool]

    if len(valid_pixels) == 0:
        print("Error: No valid disc pixels found. Check threshold values.")
        return

    # 4. Apply K-Means Clustering
    # We group the valid pixels into clusters (e.g., 3: light centre, grey edge, dark patch)
    kmeans = KMeans(n_clusters=num_color_groups, random_state=42, n_init=10)
    labels = kmeans.fit_predict(valid_pixels)

    # Find the darkest cluster (lowest average RGB value) to represent the degraded patches
    cluster_brightness = np.sum(kmeans.cluster_centers_, axis=1)
    darkest_cluster_idx = np.argmin(cluster_brightness)

    # 5. Calculate Percentage
    patch_pixel_count = np.sum(labels == darkest_cluster_idx)
    total_valid_pixels = len(valid_pixels)
    patch_percentage = (patch_pixel_count / total_valid_pixels) * 100

    # 6. Highlight the Patches
    # Create an output image and a label map to map the 1D labels back to 2D coordinates
    output_image = image_rgb.copy()
    label_map = np.full(valid_mask_bool.shape, -1, dtype=int)
    label_map[valid_mask_bool] = labels

    # Colour the degraded patches bright red ([255, 0, 0] in RGB)
    output_image[label_map == darkest_cluster_idx] = [255, 0, 0]

    # 7. Display the Results
    print(f"Total Valid Disc Pixels: {total_valid_pixels}")
    print(f"Degraded Patch Pixels: {patch_pixel_count}")
    print(f"Degradation Percentage: {patch_percentage:.2f}%\n")

    # Plot original vs highlighted image side-by-side
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Disc")
    plt.imshow(image_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Highlighted Patches ({patch_percentage:.2f}%)")
    plt.imshow(output_image)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Run the function
# Replace 'disc_image.jpg' with your actual file name
analyze_disc_degradation('disc_image.jpg', num_color_groups=3)
