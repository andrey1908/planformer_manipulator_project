import numpy as np
import cv2
from detection import detect_table_aruco, detect_table_markers_on_image_hsv
from aruco import get_aruco_corners_3d
from plane_frame import PlaneFrame
from shapely.geometry import Polygon


def calibrate_table_by_aruco(image, view, K, D, aruco_size):
    arucos = detect_table_aruco(image, view, K, D, aruco_size)
    if arucos.n != 4:
        return None, None
    corners_3d = get_aruco_corners_3d(arucos)
    table_frame = PlaneFrame.from_points(corners_3d.reshape(16, 3))
    return table_frame, corners_3d


def get_table_markers_coords_in_table_frame_by_aruco(image, view, K, D, aruco_size):
    table_frame, _ = calibrate_table_by_aruco(image, view, K, D, aruco_size)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    table_markers = detect_table_markers_on_image_hsv(hsv, view)
    assert len(table_markers) == 4
    # table_markers.shape = (4, 1, 2)

    table_markers = cv2.undistortPoints(table_markers, K, D)
    keys = table_markers.sum(axis=(1, 2))
    order = keys.argsort()
    table_markers = table_markers[order]
    table_markers[[2, 3]] = table_markers[[3, 2]]
    p = Polygon(table_markers[:, 0, :])
    if p.exterior.is_ccw:
        table_markers[[1, 3]] = table_markers[[3, 1]]
        p = Polygon(table_markers[:, 0, :])
    assert p.exterior.is_simple
    assert not p.exterior.is_ccw

    table_markers = np.dstack((table_markers, np.ones((len(table_markers), 1))))
    table_markers_3d = table_frame.intersection_with_plane(table_markers)
    table_markers_2d = table_markers_3d[:, :, :2]
    table_markers_2d = table_markers_2d.astype(np.float32)
    return table_markers_2d


def calibrate_table_by_markers(image, view, K, D, target_table_markers):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    table_markers = detect_table_markers_on_image_hsv(hsv, view)
    if len(table_markers) != 4:
        return None
    # table_markers.shape = (4, 1, 2)

    table_markers = cv2.undistortPoints(table_markers, K, D)
    keys = table_markers.sum(axis=(1, 2))
    order = keys.argsort()
    table_markers = table_markers[order]
    table_markers[[2, 3]] = table_markers[[3, 2]]
    p = Polygon(table_markers[:, 0, :])
    if p.exterior.is_ccw:
        table_markers[[1, 3]] = table_markers[[3, 1]]
        p = Polygon(table_markers[:, 0, :])
    if not p.exterior.is_simple or p.exterior.is_ccw:
        return None

    table_markers = table_markers.astype(np.float32)
    table_transform = cv2.getPerspectiveTransform(table_markers, target_table_markers)
    return table_transform
