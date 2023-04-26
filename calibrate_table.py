import numpy as np
import cv2
from aruco import detect_aruco, select_aruco_poses, select_aruco_markers, \
    PoseSelectors, get_aruco_corners_3d
from estimate_plane_frame import estimate_plane_frame
from aruco_detection_configs import aruco_dict, aruco_detection_params


def calibrate_table(image, view, K, D, aruco_size):
    assert view in ("top", "front")
    if view == "top":
        pose_selector = PoseSelectors.best
    elif view == "front":
        pose_selector = PoseSelectors.Z_axis_up

    arucos = detect_aruco(image, K=K, D=D, aruco_sizes=aruco_size,
        use_generic=True, aruco_dict=aruco_dict, params=aruco_detection_params)
    arucos = select_aruco_poses(arucos, pose_selector)
    arucos = select_aruco_markers(arucos, lambda id: id < 4)
    if arucos.n != 4:
        return None, None
    corners_3d = get_aruco_corners_3d(arucos)
    camera2table = estimate_plane_frame(corners_3d.reshape(16, 3))
    return camera2table, corners_3d
