import cv2
import numpy as np
from camera_utils import stream


def show(image):
    cv2.namedWindow('show', cv2.WINDOW_NORMAL)
    cv2.imshow('show', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def select_roi(image, full_by_default=False, window_name="select roi"):
    roi = cv2.selectROI(window_name, image, showCrosshair=False)
    cv2.destroyAllWindows()

    x, y, w, h = roi
    if (x, y, w, h) == (0, 0, 0, 0):
        if full_by_default:
            x_range = slice(0, None)
            y_range = slice(0, None)
        else:
            x_range = None
            y_range = None
    else:
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


def get_image_from_camera(camera_id, window_name="camera"):
    cam = cv2.VideoCapture(camera_id)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
    camera = lambda: {"image": cam.read()[1]}
    stream(camera, window_name=window_name)
    image = camera()
    cam.release()
    return image
