import numpy as np
import cv2
from transforms3d.quaternions import axangle2quat
from transforms3d.axangles import mat2axangle
from detection import detect_boxes_aruco, detect_red_boxes_on_image_hsv, \
    detect_blue_boxes_on_image_hsv, detect_table_markers_on_image_hsv
from estimate_plane_frame import intersection_with_XY
from shapely.geometry import Polygon


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
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    red_boxes = detect_red_boxes_on_image_hsv(hsv, view)
    blue_boxes = detect_blue_boxes_on_image_hsv(hsv, view)
    boxes = np.vstack((red_boxes, blue_boxes))

    table2camera = np.linalg.inv(camera2table)
    if len(boxes) > 0:
        points = cv2.undistortPoints(boxes, K, D)
        points = points[:, 0, :]
        points = np.hstack((points, np.ones((len(points), 1))))
        points = intersection_with_XY(points, camera2table)
        points = np.hstack((points, np.ones((len(points), 1))))
        points = np.expand_dims(points, axis=-1)
        points = np.matmul(table2camera, points)
        points = points[:, :, 0]
    else:
        points = np.empty((0, 4))

    boxes_positions = points[:, :2]
    boxes_orientations = np.tile(np.array([0., 0., 0., 1.]), (len(boxes), 1))
    return boxes_positions, boxes_orientations


def detect_boxes_visual(image, view, K, D):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    red_boxes = detect_red_boxes_on_image_hsv(hsv, view)
    blue_boxes = detect_blue_boxes_on_image_hsv(hsv, view)
    table_markers = detect_table_markers_on_image_hsv(hsv, view)
    assert(len(table_markers) == 4)

    red_boxes = cv2.undistortPoints(red_boxes, K, D)
    blue_boxes = cv2.undistortPoints(blue_boxes, K, D)
    table_markers = cv2.undistortPoints(table_markers, K, D)

    table_markers.sort(key=lambda a, b: np.sign(sum(a) - sum(b)))
    p = Polygon(table_markers[:, 0, :])
    if p.exterior.is_ccw:
        table_markers[[1, 3]] = table_markers[[3, 1]]
        p = Polygon(table_markers[:, 0, :])
    assert(p.exterior.is_simple)
    assert(not p.exterior.is_ccw)

    dst = np.array([[[0, 1]], [[1, 1]], [[1, 0]], [0, 0]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(table_markers, dst)

    boxes = np.vstack((red_boxes, blue_boxes))
    boxes_positions = cv2.perspectiveTransform(boxes, transform).squeeze()
    boxes_orientations = np.tile(np.array([0., 0., 0., 1.]), (len(boxes_positions), 1))
    return boxes_positions, boxes_orientations
