import numpy as np
from PIL import Image
import cv2


def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)


def save_image(arr, path):
    Image.fromarray(arr).save(path)


# ---------- STEP 1: Create defect mask ----------
def generate_defect_mask(shape, coverage=0.2, scale=10):
    h, w = shape[:2]

    # random noise
    noise = np.random.rand(h, w)

    # smooth it (creates blobs instead of pixels)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=scale, sigmaY=scale)

    # normalize
    noise = (noise - noise.min()) / (noise.max() - noise.min())

    # threshold based on desired coverage
    thresh = np.quantile(noise, 1 - coverage)

    mask = (noise > thresh).astype(np.uint8)

    return mask


# ---------- STEP 2: Apply unpolished texture ----------
def apply_defect(image, mask, intensity=75):
    img = image.astype(np.float32)

    # generate noise texture
    noise = np.random.normal(0, intensity, img.shape)

    # expand mask to 3 channels
    mask_3 = np.stack([mask] * 3, axis=-1)

    # apply noise only where mask == 1
    img = img + noise * mask_3

    return np.clip(img, 0, 255).astype(np.uint8)


# ---------- STEP 3: Optional edge bias ----------
def bias_to_edges(mask, strength=0.5):
    h, w = mask.shape

    # distance from center
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    dist = dist / dist.max()

    # amplify edges
    edge_weight = dist**2

    # combine
    new_mask = mask * (np.random.rand(h, w) < (edge_weight + strength))

    return new_mask.astype(np.uint8)


# ---------- MAIN ----------
image = load_image("Bild.jpg")  # load your disc image here

coverage = 0.3  # ✅ CONTROL THIS
mask = generate_defect_mask(image.shape, coverage=coverage, scale=12)

# optional: push defects toward edges (like your real part)
mask = bias_to_edges(mask, strength=0.8)

result = apply_defect(image, mask, intensity=75)

# save outputs
save_image(result, "synthetic.png")
save_image(mask * 255, "mask.png")
