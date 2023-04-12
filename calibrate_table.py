import numpy as np
import cv2
from aruco import detect_aruco, select_aruco_poses, select_aruco_markers, \
    PoseSelectors, get_aruco_corners_3d
from estimate_plane_frame import estimate_plane_frame


def calibrate_table(image, aruco_size, K, D, aruco_dict, params):
    arucos = detect_aruco(image, K=K, D=D, aruco_sizes=aruco_size,
        use_generic=True, aruco_dict=aruco_dict, params=params)
    arucos = select_aruco_poses(arucos, PoseSelectors.Z_axis_up)
    arucos = select_aruco_markers(arucos, lambda id: id < 4)
    if arucos.n != 4:
        return None, None
    corners_3d = get_aruco_corners_3d(arucos)
    camera2table = estimate_plane_frame(corners_3d.reshape(16, 3))
    return camera2table, corners_3d
