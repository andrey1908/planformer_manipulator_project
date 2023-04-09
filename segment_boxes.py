import numpy as np
import cv2


def segment_and_draw_boxes_by_aruco(image, arucos, K, D,
        aruco_margin=0.0095, box_size=0.03):
    assert arucos.n_poses == 1
    n = arucos.n
    polygons_list = list()
    for i in range(n):
        rvec = arucos.rvecs[i, 0]
        R, _ = cv2.Rodrigues(rvec)
        x = R[:, 0]
        y = R[:, 1]
        z = R[:, 2]
        center = arucos.tvecs[i, 0]

        box_corners = list()
        aruco_size = arucos.aruco_sizes[i]
        first_corner = center + \
            -x * (aruco_size / 2 + aruco_margin) + \
            y * (aruco_size / 2 + aruco_margin)
        for dx, dy, dz in [(0, 0, 0), (1, 0, 0), (1, -1, 0), (0, -1, 0),
                (0, 0, -1), (1, 0, -1), (1, -1, -1), (0, -1, -1)]:
            box_corner = np.array(
                first_corner +
                x * dx * box_size +
                y * dy * box_size +
                z * dz * box_size)
            box_corners.append(box_corner)
        box_corners = np.array(box_corners)
        # box_corners.shape = (8, 3)

        top_face = box_corners[[0, 1, 2, 3], :]
        front_face = box_corners[[3, 2, 6, 7], :]
        right_face = box_corners[[2, 1, 5, 6], :]
        left_face = box_corners[[0, 3, 7, 4], :]
        bottom_face = box_corners[[7, 6, 5, 4], :]
        back_face = box_corners[[1, 0, 4, 5], :]
        box_faces = np.stack(
            (top_face, front_face, right_face, left_face, bottom_face, back_face),
            axis=0)
        # box_faces.shape = (6, 4, 3)

        segmented = np.zeros(image.shape[:2], dtype=np.uint8)
        points, _ = \
            cv2.projectPoints(box_faces.reshape(-1, 3), np.zeros(3), np.zeros(3), K, D)
        points = points.reshape(6, 4, 2).astype(np.int)
        for pts in points:
            cv2.fillPoly(segmented, [pts], 255)

        polygons, _ = \
            cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        assert len(polygons) == 1
        polygon = polygons[0].reshape(-1, 2)
        polygons_list.append(polygon)

    cv2.polylines(image, polygons_list, True, (255, 255, 255), thickness=1)
    overlay = image.copy()
    for polygon in polygons_list:
        cv2.fillPoly(overlay, [polygon], (255, 255, 255))
    cv2.addWeighted(image, 0.7, overlay, 0.3, 0, dst=image)


def segment_boxes_by_color(image: np.ndarray):
    assert len(image.shape) == 3
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    # shift hue so that red color is continuous
    hsv = hsv + np.array([100, 0, 0], dtype=np.uint8).reshape(1, 1, 3)
    low = np.array([92, 150, 90], dtype=np.uint8)
    up = np.array([108, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, low, up)

    polygons, _ = \
        cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask[...] = 0
    for polygon in polygons:
        if len(polygon) < 100:
            continue
        cv2.fillPoly(mask, [polygon], 255)

    return mask
