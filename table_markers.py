import numpy as np
import cv2
from detection import detect_table_markers_on_image_hsv


def detect_and_rearrange_table_markers_on_image_hsv(hsv, view=""):
    table_markers, ((refined_mask, orig_mask), table_markers_polygons) = \
        detect_table_markers_on_image_hsv(hsv, view=view)
    tl_index = get_tl_table_marker_index(refined_mask, orig_mask, table_markers_polygons)
    table_markers = rearrange_table_markers(table_markers, tl_index)
    return table_markers


def get_tl_table_marker_index(refined_mask, orig_mask, table_markers_polygons):
    assert len(table_markers_polygons) > 0
    tl_index = 0
    max_diff_total_rate = 0
    for i in range(len(table_markers_polygons)):
        table_marker_mask = np.zeros_like(refined_mask)
        cv2.drawContours(table_marker_mask, table_markers_polygons, i, color=1, thickness=-1)
        refined_intersection = np.logical_and(refined_mask, table_marker_mask)
        orig_intersection = np.logical_and(orig_mask, table_marker_mask)
        refined_num = np.count_nonzero(refined_intersection)
        orig_num = np.count_nonzero(orig_intersection)
        total = np.count_nonzero(table_marker_mask)
        diff = refined_num - orig_num
        assert diff >= 0
        diff_total_rate = diff / total
        if diff_total_rate > max_diff_total_rate:
            tl_index = i
            max_diff_total_rate = diff_total_rate
    return tl_index


def rearrange_table_markers(table_markers, first_table_marker_index):
    assert len(table_markers) == 4
    # table_markers.shape = (4, 1, 2)
    assert 0 <= first_table_marker_index < 4

    center = table_markers.mean(axis=0, keepdims=True)
    # center.shape = (1, 1, 2)
    table_markers_shifted = table_markers - center
    angles = np.arctan2(table_markers_shifted[:, 0, 1], table_markers_shifted[:, 0, 0])
    order = np.argsort(angles)
    order = np.roll(order, -np.where(order == first_table_marker_index)[0][0])
    table_markers = table_markers[order]

    return table_markers


def get_table_markers_coords_in_table_frame(image, view, K, D, table_frame, markers_height):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    table_markers = detect_and_rearrange_table_markers_on_image_hsv(hsv, view)
    assert len(table_markers) == 4
    # table_markers.shape = (4, 1, 2)

    table_markers = cv2.undistortPoints(table_markers, K, D)
    table_markers_3d = np.dstack((table_markers, np.ones((len(table_markers), 1, 1))))
    table_markers_3d = table_frame.intersection_with_plane(
        table_markers_3d, shift=markers_height)
    table_markers_3d = table_frame.to_plane(table_markers_3d)
    table_markers_2d = table_markers_3d[:, :, :2]
    table_markers_2d = table_markers_2d.astype(np.float32)
    # table_markers_2d.shape = (4, 1, 2)
    return table_markers_2d
