import numpy as np
import cv2
from detection import detect_table_aruco, detect_table_markers_on_image_hsv, \
    rearrange_table_markers
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
    table_markers = rearrange_table_markers(table_markers)

    table_markers = np.dstack((table_markers, np.ones((len(table_markers), 1))))
    table_markers_3d = table_frame.intersection_with_plane(table_markers)
    table_markers_3d = table_frame.to_plane(table_markers_3d)
    table_markers_2d = table_markers_3d[:, :, :2]
    table_markers_2d = table_markers_2d.astype(np.float32)
    return table_markers_2d


def calibrate_table_by_markers(image, view, K, D, target_table_markers=None, table_aruco_size=None):
    assert (target_table_markers is not None) or (table_aruco_size is not None)
    if target_table_markers is None:
        target_table_markers = \
            get_table_markers_coords_in_table_frame_by_aruco(image, view, K, D, table_aruco_size)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    table_markers = detect_table_markers_on_image_hsv(hsv, view)
    if len(table_markers) != 4:
        return None
    # table_markers.shape = (4, 1, 2)

    table_markers = cv2.undistortPoints(table_markers, K, D)
    table_markers = rearrange_table_markers(table_markers)

    table_markers = table_markers.astype(np.float32)
    table_transform = cv2.getPerspectiveTransform(table_markers, target_table_markers)
    return table_transform
