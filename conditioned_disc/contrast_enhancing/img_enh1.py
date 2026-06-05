import cv2
import numpy as np
import os


def display_image(title, image):
    try:
        cv2.imwrite(title + ".jpg", image)
    except ImportError:
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


image_path = "../og_image_completely_masked.jpg"
image = cv2.imread(image_path)
# image_resized = cv2.resize(image, (500, 600))
image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=4)
clahe_img = np.clip(clahe.apply(image_bw) + 255, 0, 255).astype(np.uint8)
_, threshold_img = cv2.threshold(image_bw, 160, 255, cv2.THRESH_BINARY)
display_image("ordinary_threshold_160", threshold_img)
display_image("Ordinary Threshold", threshold_img)
# display_image("CLAHE_clip_255", clahe_img)
display_image("CLAHE Image", clahe_img)
