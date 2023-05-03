import numpy as np
import cv2
from params import segmentation_roi
from segment_by_color import segment_by_color


def segment_red_boxes_hsv(hsv, view=""):
    assert view in ("top", "front", "")
    # shift hue so that red color is continuous
    hsv = hsv + np.array([150, 0, 0], dtype=np.uint8).reshape(1, 1, 3)
    min_color = np.array([147 - 9, 110, 120], dtype=np.uint8)
    max_color = np.array([147 + 9, 255, 255], dtype=np.uint8)
    if view:
        x_range, y_range = segmentation_roi[view]["boxes"]
    else:
        x_range, y_range = slice(0, None), slice(0, None)
    mask, num, _ = segment_by_color(hsv, min_color, max_color,
        x_range=x_range, y_range=y_range,
        refine_mask=True, min_polygon_length=100, max_polygon_length=1000)
    return mask, num


def segment_blue_boxes_hsv(hsv, view=""):
    assert view in ("top", "front", "")
    min_color = np.array([161 - 9, 110, 120], dtype=np.uint8)
    max_color = np.array([161 + 9, 255, 255], dtype=np.uint8)
    if view:
        x_range, y_range = segmentation_roi[view]["boxes"]
    else:
        x_range, y_range = slice(0, None), slice(0, None)
    mask, num, _ = segment_by_color(hsv, min_color, max_color,
        x_range=x_range, y_range=y_range,
        refine_mask=True, min_polygon_length=100, max_polygon_length=1000)
    return mask, num


def segment_goal_hsv(hsv, view=""):
    assert view in ("top", "front", "")
    min_color = np.array([45 - 9, 60, 160], dtype=np.uint8)
    max_color = np.array([45 + 9, 255, 255], dtype=np.uint8)
    if view:
        x_range, y_range = segmentation_roi[view]["goal_and_stop_line"]
    else:
        x_range, y_range = slice(0, None), slice(0, None)
    mask, num, _ = segment_by_color(hsv, min_color, max_color,
        x_range=x_range, y_range=y_range,
        refine_mask=True, min_polygon_length=100, max_polygon_length=1000)
    return mask, num


def segment_stop_line_hsv(hsv, view=""):
    assert view in ("top", "front", "")
    min_color = np.array([0, 0, 0], dtype=np.uint8)
    max_color = np.array([255, 255, 80], dtype=np.uint8)
    if view:
        x_range, y_range = segmentation_roi[view]["goal_and_stop_line"]
    else:
        x_range, y_range = slice(0, None), slice(0, None)
    mask, num, _ = segment_by_color(hsv, min_color, max_color,
        x_range=x_range, y_range=y_range,
        refine_mask=True, min_polygon_length=100, max_polygon_length=1000)
    return mask, num
