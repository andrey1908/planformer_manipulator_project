import numpy as np
import cv2
from aruco import detect_aruco, select_aruco_poses, select_aruco_markers, PoseSelectors
from params import aruco_dict, aruco_detection_params, retry_rejected_params
from segmentation import segment_red_boxes_hsv, segment_blue_boxes_hsv


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
    arucos = select_aruco_markers(arucos, lambda id: id >= 4)
    return arucos


def detect_boxes_on_image(image, view):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    _, _, red_polygons = segment_red_boxes_hsv(hsv, view)
    _, _, blue_polygons = segment_blue_boxes_hsv(hsv, view)

    red_points = list()
    for red_polygon in red_polygons:
        u, v = red_polygon.mean(axis=0)[0]
        red_points.append(np.array([[u, v]]))

    blue_points = list()
    for blue_polygon in blue_polygons:
        u, v = blue_polygon.mean(axis=0)[0]
        blue_points.append(np.array([[u, v]]))

    points = np.array(red_points + blue_points)
    points_numbers = [("red", len(red_points)), ("blue", len(blue_points))]
    return points, points_numbers
