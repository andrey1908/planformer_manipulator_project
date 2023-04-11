import numpy as np
import cv2
from transforms3d.quaternions import axangle2quat
from transforms3d.axangles import mat2axangle
from utils.aruco import detect_aruco, select_aruco_poses, select_aruco_markers, \
    PoseSelectors


def detect_boxes(image, K, D, camera2table, caemra_position,
        aruco_size=0.016, box_size=0.03):
    assert caemra_position in ("side", "top")
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
    params = cv2.aruco.DetectorParameters_create()
    arucos = detect_aruco(image, K=K, D=D, aruco_sizes=aruco_size, use_generic=True,
        aruco_dict=aruco_dict, params=params)
    if caemra_position == "side":
        arucos = select_aruco_poses(arucos, PoseSelectors.Z_axis_up)
    elif caemra_position == "top":
        arucos = select_aruco_poses(arucos, PoseSelectors.Z_axis_back)
    arucos = select_aruco_markers(arucos, lambda id: id >= 4)

    print(f"Detected {arucos.n} boxes")
    if len(arucos.n) == 0:
        return np.empty((0, 2)), np.empty((0, 4))

    marker_poses_in_camera = np.tile(np.eye(4), (arucos.n, 1, 1))
    for i in range(arucos.n):
        marker_poses_in_camera[i, 0:3, 0:3], _ = cv2.Rodrigues(arucos.rvecs[i])
        marker_poses_in_camera[i, 0:3, 3] = arucos.tvecs[i, 0]
    # marker_poses_in_camera.shape = (n, 4, 4)

    marker_poses = np.matmul(np.linalg.inv(camera2table), marker_poses_in_camera)
    # marker_poses.shape = (n, 4, 4)

    marker2box = np.eye(4)
    marker2box[2, 3] = -box_size / 2
    boxes_poses = np.matmul(marker_poses, marker2box)
    # boxes_poses.shape = (n, 4, 4)

    boxes_positions = boxes_poses[:, 0:2, 3]
    # boxes_positions.shape = (n, 2)

    boxes_orientations = list()
    for i in range(arucos.n):
        axis, angle = mat2axangle(boxes_poses[i, 0:3, 0:3])
        assert abs(np.linalg.norm(axis) - 1.0) < 0.0001
        newAxis = np.array([0., 0., 1.])
        newAngle = angle * np.dot(axis, newAxis)
        quat = axangle2quat(newAxis, newAngle)
        quat = np.array([quat[1], quat[2], quat[3], quat[0]])
        boxes_orientations.append(quat)
    boxes_orientations = np.array(boxes_orientations)
    # boxes_orientations.shape = (n, 4)

    return boxes_positions, boxes_orientations
