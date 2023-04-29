import cv2
import numpy as np


def show(image):
    cv2.namedWindow('show', cv2.WINDOW_NORMAL)
    cv2.imshow('show', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_roi(image, window_name=""):
    roi = cv2.selectROI(window_name, image, showCrosshair=False)
    cv2.destroyAllWindows()
    x, y, w, h = roi

    x_range = slice(x, x + w)
    y_range = slice(y, y + h)

    return x_range, y_range


def get_color_range(image, window_name=""):
    roi = cv2.selectROI(window_name, image, showCrosshair=False)
    cv2.destroyAllWindows()
    x, y, w, h = roi

    sub_image = image[y: y + h, x: x + w]
    min_colors = sub_image.min(axis=(0, 1))
    max_colors = sub_image.max(axis=(0, 1))

    return min_colors, max_colors
