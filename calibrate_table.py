import numpy as np
import cv2
from detection import detect_table_aruco
from aruco import detect_aruco, select_aruco_poses, select_aruco_markers, \
    PoseSelectors, get_aruco_corners_3d
from estimate_plane_frame import estimate_plane_frame
from aruco_detection_configs import aruco_dict, aruco_detection_params


def calibrate_table(image, view, K, D, aruco_size):
    arucos = detect_table_aruco(image, view, K, D, aruco_size)
    if arucos.n != 4:
        return None, None
    corners_3d = get_aruco_corners_3d(arucos)
    camera2table = estimate_plane_frame(corners_3d.reshape(16, 3))
    return camera2table, corners_3d
