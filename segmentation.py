import numpy as np
import cv2
from params import segmentation_roi
from kas_utils.segment_by_color import get_mask_in_roi, refine_mask_by_polygons


def segment_red_boxes_hsv(hsv, view=""):
    assert view in ("top", "front", "")
    # shift hue so that red color is continuous
    hsv = hsv + np.array([100, 0, 0], dtype=np.uint8)
    min_color = np.array([63, 120, 50], dtype=np.uint8)
    max_color = np.array([106, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, min_color, max_color)
    if view:
        x_range, y_range = segmentation_roi[view]["working_area"]
        mask = get_mask_in_roi(mask, x_range, y_range)
    mask, polygons = refine_mask_by_polygons(mask,
        min_polygon_length=80, max_polygon_length=250,
        min_polygon_area_length_ratio=5)
    return mask, polygons


def segment_blue_boxes_hsv(hsv, view=""):
    assert view in ("top", "front", "")
    min_color = np.array([132, 120, 50], dtype=np.uint8)
    max_color = np.array([186, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, min_color, max_color)
    if view:
        x_range, y_range = segmentation_roi[view]["working_area"]
        mask = get_mask_in_roi(mask, x_range, y_range)
    mask, polygons = refine_mask_by_polygons(mask,
        min_polygon_length=80, max_polygon_length=250,
        min_polygon_area_length_ratio=5)
    return mask, polygons


def segment_goal_hsv(hsv, view=""):
    assert view in ("top", "front", "")
    min_color = np.array([36, 50, 150], dtype=np.uint8)
    max_color = np.array([68, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, min_color, max_color)
    if view:
        x_range, y_range = segmentation_roi[view]["goal_and_stop_line"]
        mask = get_mask_in_roi(mask, x_range, y_range)
    mask, polygons = refine_mask_by_polygons(mask,
        min_polygon_length=150, max_polygon_length=1500,
        select_top_n_polygons_by_length=3)
    return mask, polygons


def segment_stop_line_hsv(hsv, view=""):
    assert view in ("top", "front", "")
    min_color = np.array([90, 60, 80], dtype=np.uint8)
    max_color = np.array([130, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, min_color, max_color)
    if view:
        x_range, y_range = segmentation_roi[view]["goal_and_stop_line"]
        mask = get_mask_in_roi(mask, x_range, y_range)
    mask, polygons = refine_mask_by_polygons(mask,
        min_polygon_length=400, max_polygon_length=3000,
        select_top_n_polygons_by_length=1)
    return mask, polygons


def segment_table_markers_hsv(hsv, view=""):
    assert view in ("top", "front", "")
    min_color = np.array([90, 60, 80], dtype=np.uint8)
    max_color = np.array([130, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, min_color, max_color)
    if view:
        x_range, y_range = segmentation_roi[view]["working_area"]
        mask = get_mask_in_roi(mask, x_range, y_range)
    refined_mask, polygons = refine_mask_by_polygons(mask,
        min_polygon_length=30, max_polygon_length=150)
    return (refined_mask, mask), polygons
