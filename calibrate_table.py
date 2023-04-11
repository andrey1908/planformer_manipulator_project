import numpy as np
import cv2
from utils.aruco import detect_aruco, select_aruco_poses, select_aruco_markers, \
    PoseSelectors, get_aruco_corners_3d
from utils.estimate_plane_frame import estimate_plane_frame


def calibrate_table(image, aruco_size, K, D, caemra_position):
    assert caemra_position in ("side", "top")
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
    params = cv2.aruco.DetectorParameters_create()
    arucos = detect_aruco(image, K=K, D=D, aruco_sizes=aruco_size,
        use_generic=True, aruco_dict=aruco_dict, params=params)
    if caemra_position == "side":
        arucos = select_aruco_poses(arucos, PoseSelectors.Z_axis_up)
    elif caemra_position == "top":
        arucos = select_aruco_poses(arucos, PoseSelectors.Z_axis_back)
    arucos = select_aruco_markers(arucos, lambda id: id < 4)
    if len(arucos.n) != 4:
        return None, None
    corners_3d = get_aruco_corners_3d(arucos).reshape(16, 3)
    camera2table = estimate_plane_frame(corners_3d)
    return camera2table, corners_3d
