import numpy as np
import cv2
from params import segmentation_roi
from segment_by_color import segment_by_color


def segment_red_boxes_hsv(hsv, view=""):
    assert view in ("top", "front", "")
    # shift hue so that red color is continuous
    hsv = hsv + np.array([100, 0, 0], dtype=np.uint8)
    min_color = np.array([63, 110, 90], dtype=np.uint8)
    max_color = np.array([106, 255, 255], dtype=np.uint8)
    if view:
        x_range, y_range = segmentation_roi[view]["working_area"]
    else:
        x_range, y_range = slice(0, None), slice(0, None)
    mask, polygons = segment_by_color(hsv, min_color, max_color,
        x_range=x_range, y_range=y_range,
        refine=True, min_polygon_length=80, max_polygon_length=250)
    return mask, polygons


def segment_blue_boxes_hsv(hsv, view=""):
    assert view in ("top", "front", "")
    min_color = np.array([132, 110, 90], dtype=np.uint8)
    max_color = np.array([186, 255, 255], dtype=np.uint8)
    if view:
        x_range, y_range = segmentation_roi[view]["working_area"]
    else:
        x_range, y_range = slice(0, None), slice(0, None)
    mask, polygons = segment_by_color(hsv, min_color, max_color,
        x_range=x_range, y_range=y_range,
        refine=True, min_polygon_length=80, max_polygon_length=250)
    return mask, polygons


def segment_goal_hsv(hsv, view=""):
    assert view in ("top", "front", "")
    min_color = np.array([36, 50, 150], dtype=np.uint8)
    max_color = np.array([68, 255, 255], dtype=np.uint8)
    if view:
        x_range, y_range = segmentation_roi[view]["goal_and_stop_line"]
    else:
        x_range, y_range = slice(0, None), slice(0, None)
    mask, polygons = segment_by_color(hsv, min_color, max_color,
        x_range=x_range, y_range=y_range,
        refine=True, min_polygon_length=150, max_polygon_length=1500)
    return mask, polygons


def segment_stop_line_hsv(hsv, view=""):
    assert view in ("top", "front", "")
    min_color = np.array([0, 0, 0], dtype=np.uint8)
    max_color = np.array([255, 255, 100], dtype=np.uint8)
    if view:
        x_range, y_range = segmentation_roi[view]["goal_and_stop_line"]
    else:
        x_range, y_range = slice(0, None), slice(0, None)
    mask, polygons = segment_by_color(hsv, min_color, max_color,
        x_range=x_range, y_range=y_range,
        refine=True, min_polygon_length=500, max_polygon_length=900)
    return mask, polygons


def segment_table_markers_hsv(hsv, view=""):
    assert view in ("top", "front", "")
    min_color = np.array([90, 60, 80], dtype=np.uint8)
    max_color = np.array([130, 255, 255], dtype=np.uint8)
    if view:
        x_range, y_range = segmentation_roi[view]["working_area"]
    else:
        x_range, y_range = slice(0, None), slice(0, None)
    (refined_mask, orig_mask), polygons = segment_by_color(hsv, min_color, max_color,
        x_range=x_range, y_range=y_range,
        refine=True, min_polygon_length=30, max_polygon_length=150,
        return_orig_mask=True)
    return (refined_mask, orig_mask), polygons
