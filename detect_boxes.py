import numpy as np
import cv2
from transforms3d.quaternions import axangle2quat
from transforms3d.axangles import mat2axangle
from detection import detect_boxes_aruco, detect_boxes_on_image
from estimate_plane_frame import intersection_with_XY
from segmentation import segment_red_boxes_hsv, segment_blue_boxes_hsv, \
    segment_green_markers_hsv
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
    image_points, points_numbers = detect_boxes_on_image(image, view)

    table2camera = np.linalg.inv(camera2table)
    if len(image_points) > 0:
        points = cv2.undistortPoints(image_points, K, D)
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
    boxes_orientations = np.tile(np.array([0., 0., 0., 1.]), (len(boxes_positions), 1))
    return boxes_positions, boxes_orientations


def detect_boxes_visual(image, view, K, D):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    _, _, red_polygons = segment_red_boxes_hsv(hsv, view)
    _, _, blue_polygons = segment_blue_boxes_hsv(hsv, view)
    _, _, green_polygons = segment_green_markers_hsv(hsv, view)
    assert(len(green_polygons) == 4)

    red_boxes = list()
    for red_polygon in red_polygons:
        u, v = red_polygon.mean(axis=0)[0]
        red_boxes.append((u, v))
    red_boxes = np.array(red_boxes)[:, np.newaxis, :]
    red_boxes = cv2.undistortPoints(red_boxes, K, D)

    blue_boxes = list()
    for blue_polygon in blue_polygons:
        u, v = blue_polygon.mean(axis=0)[0]
        blue_boxes.append((u, v))
    blue_boxes = np.array(blue_boxes)[:, np.newaxis, :]
    blue_boxes = cv2.undistortPoints(blue_boxes, K, D)

    green_markers = list()
    for green_polygon in green_polygons:
        u, v = green_polygon.mean(axis=0)[0]
        green_markers.append((u, v))
    green_markers.sort(key=lambda a, b: np.sign(sum(a) - sum(b)))
    p = Polygon(green_markers)
    if p.exterior.is_ccw:
        green_markers[1], green_markers[3] = green_markers[3], green_markers[1]
        p = Polygon(green_markers)
    assert(p.exterior.is_simple)
    assert(not p.exterior.is_ccw)

    green_markers = np.array(green_markers)[:, np.newaxis, :]
    green_markers = cv2.undistortPoints(green_markers, K, D)
    dst = np.array([[[0, 1]], [[1, 1]], [[1, 0]], [0, 0]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(green_markers, dst)

    red_poses = cv2.perspectiveTransform(red_boxes, transform).squeeze()
    blue_poses = cv2.perspectiveTransform(blue_boxes, transform).squeeze()

    boxes_positions = np.vstack((red_poses, blue_poses))
    boxes_orientations = np.tile(np.array([0., 0., 0., 1.]), (len(boxes_positions), 1))
    return boxes_positions, boxes_orientations
