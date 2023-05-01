import cv2
import numpy as np
from aruco import RetryRejectedParameters


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


aruco_dict = get_aruco_dict()
aruco_detection_params = get_aruco_detection_params()
retry_rejected_params = get_retry_rejected_params()

K, D = get_camera_calib('data/calib.npz')

table_aruco_size = 0.132
box_aruco_size = 0.0172
box_size = 0.03