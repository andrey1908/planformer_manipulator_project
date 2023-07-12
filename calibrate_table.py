import numpy as np
import cv2
from detection import detect_table_aruco
from table_markers import detect_and_rearrange_table_markers_on_image_hsv
from aruco import get_aruco_corners_3d
from plane_frame import PlaneFrame


def calibrate_table_by_aruco(image, view, K, D, aruco_size):
    arucos = detect_table_aruco(image, view, K, D, aruco_size)
    if arucos.n != 4:
        return None, None
    corners_3d_in_camera = get_aruco_corners_3d(arucos)
    table_frame = PlaneFrame.from_points(corners_3d_in_camera.reshape(16, 3))
    return table_frame, corners_3d_in_camera


def calibrate_table_by_markers(image, view, K, D, target_table_markers):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    table_markers = detect_and_rearrange_table_markers_on_image_hsv(hsv, view=view)
    if len(table_markers) != 4:
        return None
    # table_markers.shape = (4, 1, 2)

    table_markers = cv2.undistortPoints(table_markers, K, D)

    table_markers = table_markers.astype(np.float32)
    target_table_markers = target_table_markers.astype(np.float32)
    table_transform = cv2.getPerspectiveTransform(table_markers, target_table_markers)
    return table_transform
