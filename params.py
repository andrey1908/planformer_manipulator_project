import cv2
import numpy as np
from kas_utils.aruco import RetryRejectedParameters
import torch
from ultralytics import YOLO
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
    with open(segmentation_roi_file, 'rb') as f:
        segmentation_roi = pickle.load(f)
    return segmentation_roi


def get_target_table_markers(target_table_markers_file):
    target_table_markers = np.load(target_table_markers_file)
    target_table_markers = target_table_markers.astype(np.float32)
    return target_table_markers


aruco_dict = get_aruco_dict()
aruco_detection_params = get_aruco_detection_params()
retry_rejected_params = get_retry_rejected_params()

K, D = get_camera_calib(osp.join(osp.dirname(__file__), "data/top_calib.npz"))

table_aruco_size = 0.132
box_aruco_size = 0.0172
box_size = 0.03

table_aruco_dist_0_1 = 29.9
table_aruco_dist_1_3 = 79.6
table_aruco_dist_3_2 = 30.0
table_aruco_dist_2_0 = 79.5

segmentation_roi = get_segmentation_roi(osp.join(osp.dirname(__file__), "data/segmentation_roi.pickle"))

target_table_markers = get_target_table_markers(osp.join(osp.dirname(__file__), "data/target_table_markers.npy"))

top_camera_id = 0
front_camera_id = 4

use_nn = True
yolov8n_model = "data/yolov8n.yaml"
yolov8n_weights = "data/yolov8n.pt"
if use_nn:
    assert yolov8n_model and yolov8n_weights
    yolov8n = YOLO(yolov8n_model)
    weights = torch.load(yolov8n_weights)['model']
    yolov8n.model.load(weights)
    yolov8n.warmup()
else:
    yolov8n = None
