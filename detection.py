import numpy as np
import cv2
from transforms3d.quaternions import axangle2quat
from transforms3d.axangles import mat2axangle
from aruco import detect_aruco, select_aruco_poses, select_aruco_markers, PoseSelectors
from aruco_detection_configs import aruco_dict, aruco_detection_params, retry_rejected_params
from segmentation import segment_red_boxes_hsv, segment_blue_boxes_hsv
from estimate_plane_frame import intersection_with_XY


def detect_boxes(image, view, K, D, camera2table, aruco_size, box_size):
    arucos = detect_boxes_aruco(image, view, K, D, aruco_size)
    print(f"Detected {arucos.n} boxes")
    if arucos.n == 0:
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


def detect_boxes_segm(image, view, K, D, camera2table, box_size):
    image_points, points_numbers = detect_boxes_on_image(image, view)

    table2camera = np.linalg.inv(camera2table)
    if len(image_points) > 0:
        points = cv2.undistortPoints(image_points, K, D)
        points = points[:, 0, :]
        points = np.hstack((points, np.ones((len(points), 1.))))
        points = intersection_with_XY(points, camera2table)
        points = np.hstack((points, np.ones((len(points), 1.))))
        points = np.expand_dims(points, axis=-1)
        points = np.matmul(table2camera, points)
        points = points[:, :, 0]
    else:
        points = np.empty((0, 4))

    boxes_positions = points[:, :2]
    boxes_orientations = np.tile(np.array([0., 0., 0., 1.]), (len(boxes_positions), 1))
    return boxes_positions, boxes_orientations


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
    red_mask, _ = segment_red_boxes_hsv(hsv)
    blue_mask, _ = segment_blue_boxes_hsv(hsv, view)

    red_polygons, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    blue_polygons, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

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
