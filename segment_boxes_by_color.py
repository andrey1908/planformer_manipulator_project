import numpy as np
import cv2


def segment_boxes_by_color(image: np.ndarray):
    assert len(image.shape) == 3
    image = image.astype(np.float64)
    intensity = np.expand_dims(np.sum(image, axis=2), axis=-1)
    normalized = image / intensity
    red = normalized[:, :, 2]
    red *= (image[:, :, 2].astype(np.float64) / 255)
    red = (red * 255).astype(np.uint8)
    return red


def segment_boxes_by_color_hsv(image: np.ndarray):
    assert len(image.shape) == 3
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    # shift hue so that red color is continuous
    hsv = hsv + np.array([100, 0, 0], dtype=np.uint8).reshape(1, 1, 3)
    low = np.array([92, 170, 60], dtype=np.uint8)
    up = np.array([108, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, low, up)
    return mask
