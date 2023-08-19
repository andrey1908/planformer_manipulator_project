import numpy as np
import cv2
from segmentation import segment_red_boxes_hsv, segment_blue_boxes_hsv, \
    segment_goal_hsv, segment_stop_line_hsv, segmnet_nn
from params import use_nn


def segment_scene_colorful(image, view=""):
    if not use_nn:
        assert len(image.shape) == 3
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)

        red_mask, red_polygons = segment_red_boxes_hsv(hsv, view=view)
        blue_mask, blue_polygons = segment_blue_boxes_hsv(hsv, view=view)
        goal_mask, goal_polygons = segment_goal_hsv(hsv, view=view)
        stop_line_mask, stop_line_polygons = segment_stop_line_hsv(hsv, view=view)
    else:
        red_mask, blue_mask, goal_mask, stop_line_mask, \
            red_polygons, blue_polygons, goal_polygons, stop_line_polygons = \
                segmnet_nn(image)

    num_red = len(red_polygons)
    num_blue = len(blue_polygons)
    num_goals = len(goal_polygons)
    num_stop_lines = len(stop_line_polygons)

    # assert num_goals == 3
    # assert num_stop_lines == 1

    segmentation = np.zeros(image.shape, dtype=image.dtype)
    segmentation[:, :, 2] = np.maximum(segmentation[:, :, 2], red_mask)
    segmentation[:, :, 0] = np.maximum(segmentation[:, :, 0], blue_mask)
    segmentation[:, :, 1] = np.maximum(segmentation[:, :, 1], goal_mask)
    segmentation[:, :, 2] = np.maximum(segmentation[:, :, 2], goal_mask)
    segmentation[:, :, 1] = np.maximum(segmentation[:, :, 1], stop_line_mask)
    segmentation[segmentation == 0] = 60
    return segmentation, (num_red, num_blue)


def segment_scene(image, view=""):
    assert len(image.shape) == 3
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)

    red_mask, red_polygons = segment_red_boxes_hsv(hsv, view=view)
    blue_mask, blue_polygons = segment_blue_boxes_hsv(hsv, view=view)
    goal_mask, goal_polygons = segment_goal_hsv(hsv, view=view)
    stop_line_mask, stop_line_polygons = segment_stop_line_hsv(hsv, view=view)
    num_red = len(red_polygons)
    num_blue = len(blue_polygons)
    num_goals = len(goal_polygons)
    num_stop_lines = len(stop_line_polygons)

    assert num_goals == 3
    assert num_stop_lines == 1

    segmentation = np.ones(image.shape[:2], dtype=image.dtype)
    segmentation[goal_mask == 255] = 4
    segmentation[stop_line_mask == 255] = 5
    segmentation[red_mask == 255] = 6
    segmentation[blue_mask == 255] = 7

    segmentation = cv2.resize(segmentation, (256, 128), interpolation=cv2.INTER_NEAREST)
    return segmentation, (num_red, num_blue)
