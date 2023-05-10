import cv2
import numpy as np
from aruco import RetryRejectedParameters
import pickle
import os.path as osp


def get_aruco_dict():
    dict_4x4 = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
    return dict_4x4


def get_aruco_detection_params():
    params = cv2.aruco.DetectorParameters_create()
    params.adaptiveThreshWinSizeMin = 8
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 5
    params.perspectiveRemovePixelPerCell = 8
    params.perspectiveRemoveIgnoredMarginPerCell = 0.26
    return params


def get_retry_rejected_params():
    retry_rejected_params = RetryRejectedParameters()
    retry_rejected_params.add_retried_areas_to_rejected = True
    return retry_rejected_params


def get_camera_calib(calib_file):
    calib = np.load(calib_file)
    K = calib['K']
    D = calib['D']
    return K, D


def get_segmentation_roi(segmentation_roi_file):
    if osp.isfile(segmentation_roi_file):
        with open(segmentation_roi_file, 'rb') as f:
            segmentation_roi = pickle.load(f)
    else:
        full_image_roi = (slice(0, None), slice(0, None))
        top_roi = {"working_area": full_image_roi, "goal_and_stop_line": full_image_roi}
        front_roi = {"working_area": full_image_roi, "goal_and_stop_line": full_image_roi}
        segmentation_roi = {"top": top_roi, "front": front_roi}
    return segmentation_roi


def get_target_table_markers(target_table_markers_file):
    target_table_markers = np.load(target_table_markers_file)
    target_table_markers = target_table_markers.astype(np.float32)
    return target_table_markers


aruco_dict = get_aruco_dict()
aruco_detection_params = get_aruco_detection_params()
retry_rejected_params = get_retry_rejected_params()

K, D = get_camera_calib(osp.join(osp.dirname(__file__), "data/calib.npz"))

table_aruco_size = 0.132
box_aruco_size = 0.0172
box_size = 0.03

table_aruco_dist_0_1 = 29.9
table_aruco_dist_1_3 = 79.6
table_aruco_dist_3_2 = 30.0
table_aruco_dist_2_0 = 79.5

segmentation_roi = get_segmentation_roi(osp.join(osp.dirname(__file__), "data/segmentation_roi.pickle"))

target_table_markers = get_target_table_markers(osp.join(osp.dirname(__file__), "data/target_table_markers.npy"))
