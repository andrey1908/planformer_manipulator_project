import numpy as np
import cv2
from params import segmentation_roi


def segment_and_draw_boxes_by_aruco(draw, arucos, K, D,
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

        segmented = np.zeros(draw.shape[:2], dtype=np.uint8)
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

    cv2.polylines(draw, polygons_list, True, (255, 255, 255), thickness=1)
    overlay = draw.copy()
    for polygon in polygons_list:
        cv2.fillPoly(overlay, [polygon], (255, 255, 255))
    cv2.addWeighted(draw, 0.7, overlay, 0.3, 0, dst=draw)


def segment_scene_colorful(image, view=""):
    assert len(image.shape) == 3
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)

    red_mask, num_red = segment_red_boxes_hsv(hsv, view)
    blue_mask, num_blue = segment_blue_boxes_hsv(hsv, view)
    goal_mask, num_goals = segment_goal_hsv(hsv, view)
    stop_line_mask, num_stop_lines = segment_stop_line_hsv(hsv, view)

    # assert num_goals == 3
    # assert num_stop_lines == 1

    segmentation = np.zeros(image.shape, dtype=image.dtype)
    segmentation[:, :, 2] = np.maximum(segmentation[:, :, 2], red_mask * 200)
    segmentation[:, :, 0] = np.maximum(segmentation[:, :, 0], blue_mask * 200)
    segmentation[:, :, 1] = np.maximum(segmentation[:, :, 1], goal_mask * 200)
    segmentation[:, :, 2] = np.maximum(segmentation[:, :, 2], goal_mask * 200)
    segmentation[:, :, 1] = np.maximum(segmentation[:, :, 1], stop_line_mask * 200)
    segmentation[segmentation == 0] = 60
    return segmentation, (num_red, num_blue)


def segment_scene(image, view=""):
    assert len(image.shape) == 3
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)

    red_mask, num_red = segment_red_boxes_hsv(hsv, view)
    blue_mask, num_blue = segment_blue_boxes_hsv(hsv, view)
    goal_mask, num_goals = segment_goal_hsv(hsv, view)
    stop_line_mask, num_stop_lines = segment_stop_line_hsv(hsv, view)

    assert num_goals == 3
    assert num_stop_lines == 1

    segmentation = np.ones(image.shape[:2], dtype=image.dtype)
    segmentation[goal_mask == 1] = 4
    segmentation[stop_line_mask == 1] = 5
    segmentation[red_mask == 1] = 6
    segmentation[blue_mask == 1] = 7

    segmentation = cv2.resize(segmentation, (256, 128), interpolation=cv2.INTER_NEAREST)
    return segmentation, (num_red, num_blue)


def segment_by_color(image, min_color, max_color, \
        x_range=slice(0, None), y_range=slice(0, None),
        refine_mask=False, min_polygon_length=100, max_polygon_length=1000):
    mask_full = cv2.inRange(image, min_color, max_color)
    mask = np.zeros(mask_full.shape, dtype=mask_full.dtype)
    mask[y_range, x_range] = mask_full[y_range, x_range]
    if refine_mask:
        polygons, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        refined_mask = np.zeros(mask.shape, dtype=mask.dtype)
        num = 0
        out_polygons = list()
        for polygon in polygons:
            if min_polygon_length <= len(polygon) <= max_polygon_length:
                cv2.fillPoly(refined_mask, [polygon], 1)
                out_polygons.append(polygon)
                num += 1
        return refined_mask, num, out_polygons
    else:
        return mask


def segment_red_boxes_hsv(hsv, view=""):
    assert view in ("top", "front", "")
    # shift hue so that red color is continuous
    hsv = hsv + np.array([150, 0, 0], dtype=np.uint8).reshape(1, 1, 3)
    min_color = np.array([147 - 9, 110, 120], dtype=np.uint8)
    max_color = np.array([147 + 9, 255, 255], dtype=np.uint8)
    if view:
        x_range, y_range = segmentation_roi[view]["boxes"]
    else:
        x_range, y_range = slice(0, None), slice(0, None)
    mask, num, _ = segment_by_color(hsv, min_color, max_color,
        x_range=x_range, y_range=y_range,
        refine_mask=True, min_polygon_length=100, max_polygon_length=1000)
    return mask, num


def segment_blue_boxes_hsv(hsv, view=""):
    assert view in ("top", "front", "")
    min_color = np.array([161 - 9, 110, 120], dtype=np.uint8)
    max_color = np.array([161 + 9, 255, 255], dtype=np.uint8)
    if view:
        x_range, y_range = segmentation_roi[view]["boxes"]
    else:
        x_range, y_range = slice(0, None), slice(0, None)
    mask, num, _ = segment_by_color(hsv, min_color, max_color,
        x_range=x_range, y_range=y_range,
        refine_mask=True, min_polygon_length=100, max_polygon_length=1000)
    return mask, num


def segment_goal_hsv(hsv, view=""):
    assert view in ("top", "front", "")
    min_color = np.array([45 - 9, 60, 160], dtype=np.uint8)
    max_color = np.array([45 + 9, 255, 255], dtype=np.uint8)
    if view:
        x_range, y_range = segmentation_roi[view]["goal_and_stop_line"]
    else:
        x_range, y_range = slice(0, None), slice(0, None)
    mask, num, _ = segment_by_color(hsv, min_color, max_color,
        x_range=x_range, y_range=y_range,
        refine_mask=True, min_polygon_length=100, max_polygon_length=1000)
    return mask, num


def segment_stop_line_hsv(hsv: np.ndarray, view: str=""):
    assert view in ("top", "front", "")
    min_color = np.array([0, 0, 0], dtype=np.uint8)
    max_color = np.array([255, 255, 80], dtype=np.uint8)
    if view:
        x_range, y_range = segmentation_roi[view]["goal_and_stop_line"]
    else:
        x_range, y_range = slice(0, None), slice(0, None)
    mask, num, _ = segment_by_color(hsv, min_color, max_color,
        x_range=x_range, y_range=y_range,
        refine_mask=True, min_polygon_length=100, max_polygon_length=1000)
    return mask, num
