import numpy as np
import cv2
from transforms3d.quaternions import axangle2quat
from transforms3d.axangles import mat2axangle
from detection import detect_boxes_aruco, detect_red_boxes_on_image_hsv, \
    detect_blue_boxes_on_image_hsv
from kas_utils.plane_frame import PlaneFrame


def detect_boxes_by_aruco(image, view, K, D, table_frame, aruco_size, box_size,
        use_intersection=True):
    arucos = detect_boxes_aruco(image, view, K, D, aruco_size)
    if arucos.n == 0:
        return np.empty((0, 2)), np.empty((0, 4))

    marker_poses_in_camera = np.tile(np.eye(4), (arucos.n, 1, 1))
    for i in range(arucos.n):
        marker_poses_in_camera[i, 0:3, 0:3], _ = cv2.Rodrigues(arucos.rvecs[i])
        marker_poses_in_camera[i, 0:3, 3] = arucos.tvecs[i, 0]
    # marker_poses_in_camera.shape = (n, 4, 4)

    marker_poses = table_frame.to_plane(marker_poses_in_camera, is_poses=True)
    # marker_poses.shape = (n, 4, 4)

    marker2box = np.eye(4)
    marker2box[2, 3] = -box_size / 2
    boxes_poses = np.matmul(marker_poses, marker2box)
    # boxes_poses.shape = (n, 4, 4)

    if use_intersection:
        if arucos.n > 0:
            aruco_centers = np.mean(arucos.corners, axis=2)
            points = cv2.undistortPoints(aruco_centers, K, D)
            points = points[:, 0, :]
            points = np.hstack((points, np.ones((len(points), 1))))
            points = table_frame.intersection_with_plane(points, shift=box_size)
            points = table_frame.to_plane(points)
            points[:, 2] -= box_size / 2
        else:
            points = np.empty((0, 4))
        boxes_positions = points[:, :2]
    else:
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


def detect_boxes_by_segmentation(image, view, K, D, table_frame, box_size):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    red_boxes = detect_red_boxes_on_image_hsv(hsv, view=view)
    blue_boxes = detect_blue_boxes_on_image_hsv(hsv, view=view)
    boxes = np.vstack((red_boxes, blue_boxes))

    if len(boxes) > 0:
        points = cv2.undistortPoints(boxes, K, D)
        points = points[:, 0, :]
        points = np.hstack((points, np.ones((len(points), 1))))
        points = table_frame.intersection_with_plane(points, shift=box_size / 2)
        points = table_frame.to_plane(points)
    else:
        points = np.empty((0, 3))

    boxes_positions = points[:, :2]
    # boxes_positions.shape = (n, 2)
    boxes_orientations = np.tile(np.array([0., 0., 0., 1.]), (len(boxes), 1))
    # boxes_orientations.shape = (n, 4)
    return boxes_positions, boxes_orientations


def detect_boxes_visual(image, view, K, D, table_transform):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    red_boxes = detect_red_boxes_on_image_hsv(hsv, view=view)
    blue_boxes = detect_blue_boxes_on_image_hsv(hsv, view=view)
    boxes = np.vstack((red_boxes, blue_boxes))

    if len(boxes) > 0:
        boxes = cv2.undistortPoints(boxes, K, D)
        boxes_positions = cv2.perspectiveTransform(boxes, table_transform)
        boxes_positions = boxes_positions[:, 0, :]
    else:
        boxes_positions = np.empty((0, 2))
    # boxes_positions.shape = (n, 2)

    boxes_orientations = np.tile(np.array([0., 0., 0., 1.]), (len(boxes_positions), 1))
    # boxes_orientations.shape = (n, 4)
    return boxes_positions, boxes_orientations
