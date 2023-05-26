import numpy as np
import cv2
from aruco import detect_aruco, select_aruco_poses, select_aruco_markers, PoseSelectors
from params import aruco_dict, aruco_detection_params, retry_rejected_params
from segmentation import segment_table_markers_hsv, segment_red_boxes_hsv, segment_blue_boxes_hsv
from shapely.geometry import Polygon


def detect_table_aruco(image, view, K, D, aruco_size):
    assert view in ("top", "front")
    if view == "top":
        pose_selector = PoseSelectors.best
    elif view == "front":
        pose_selector = PoseSelectors.Z_axis_up

    arucos = detect_aruco(image, K=K, D=D, aruco_sizes=aruco_size,
        use_generic=True, aruco_dict=aruco_dict, params=aruco_detection_params)
    arucos = select_aruco_poses(arucos, pose_selector)
    arucos = select_aruco_markers(arucos, lambda id: id < 4)
    return arucos


def detect_boxes_aruco(image, view, K, D, aruco_size):
    assert view in ("top", "front")
    if view == "top":
        pose_selector = PoseSelectors.best
    elif view == "front":
        pose_selector = PoseSelectors.Z_axis_up

    arucos = detect_aruco(image, K=K, D=D, aruco_sizes=aruco_size, use_generic=True,
        retry_rejected=True, retry_rejected_params=retry_rejected_params,
        aruco_dict=aruco_dict, params=aruco_detection_params)
    arucos = select_aruco_poses(arucos, pose_selector)
    arucos = select_aruco_markers(arucos, lambda id: id >= 4 and id <= 11)
    return arucos


def detect_red_boxes_on_image_hsv(hsv, view=""):
    _, red_boxes_polygons = segment_red_boxes_hsv(hsv, view=view)
    if len(red_boxes_polygons) > 0:
        red_boxes = get_centers_of_polygons(red_boxes_polygons)
    else:
        red_boxes = np.empty((0, 1, 2))
    # red_boxes.shape = (n, 1, 2)
    return red_boxes


def detect_blue_boxes_on_image_hsv(hsv, view=""):
    _, blue_boxes_polygons = segment_blue_boxes_hsv(hsv, view=view)
    if len(blue_boxes_polygons) > 0:
        blue_boxes = get_centers_of_polygons(blue_boxes_polygons)
    else:
        blue_boxes = np.empty((0, 1, 2))
    # blue_boxes.shape = (n, 1, 2)
    return blue_boxes


def detect_table_markers_on_image_hsv(hsv, view=""):
    (refined_mask, orig_mask), table_markers_polygons = \
        segment_table_markers_hsv(hsv, view=view)
    if len(table_markers_polygons) > 0:
        table_markers = get_centers_of_polygons(table_markers_polygons)
    else:
        table_markers = np.empty((0, 1, 2))
    # table_markers.shape = (n, 1, 2)
    return table_markers, ((refined_mask, orig_mask), table_markers_polygons)


def get_centers_of_polygons(polygons):
    centers = list()
    for polygon in polygons:
        u, v = polygon.mean(axis=0)[0]
        centers.append([(u, v)])
    centers = np.array(centers)
    # centers.shape = (n, 1, 2)
    return centers
