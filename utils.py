import cv2
import numpy as np
from aruco import detect_aruco, draw_aruco, select_aruco_poses, get_aruco_corners_3d, \
    PoseSelectors, select_aruco_markers
from calibrate_table import calibrate_table
from camera_utils import stream, StreamCallbacks
from realsense_camera import RealsenseCamera
from segment_boxes import segment_boxes_by_color


def show(image):
    cv2.namedWindow('show', cv2.WINDOW_NORMAL)
    cv2.imshow('show', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def stream_table_frame(camera, K, D, aruco_size, aruco_dict, params, save_folder=None):
    def calibrate_and_draw_table_frame(image, key):
        camera2table, _ = calibrate_table(image, K, D, aruco_size, aruco_dict, params)
        if camera2table is None:
            return
        rvec, _ = cv2.Rodrigues(camera2table[0:3, 0:3])
        tvec = camera2table[0:3, 3]
        cv2.drawFrameAxes(image, K, D, rvec, tvec, 0.1)

    if save_folder:
        save_callback = StreamCallbacks.get_save_by_key_callback(save_folder)
    else:
        save_callback = lambda: None
    stream(camera, [save_callback, calibrate_and_draw_table_frame], "stream table frame")


def stream_segmented_boxes(camera, save_folder=None):
    def segment_and_draw_boxes(image, key):
        mask, _ = segment_boxes_by_color(image)
        polygons, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.polylines(image, polygons, True, (255, 255, 255), thickness=1)
        overlay = image.copy()
        for polygon in polygons:
            cv2.fillPoly(overlay, [polygon], (255, 255, 255))
        cv2.addWeighted(image, 0.7, overlay, 0.3, 0, dst=image)

    if save_folder:
        save_callback = StreamCallbacks.get_save_by_key_callback(save_folder)
    else:
        save_callback = lambda: None
    stream(camera, [save_callback, segment_and_draw_boxes], "stream segmented boxes")


def stream_aruco_detected_on_boxes(camera, K, D, aruco_size, aruco_dict, params,
        save_folder=None):
    def detect_and_draw_aruco_on_boxes(image, key):
        arucos = detect_aruco(image, K=K, D=D, aruco_sizes=aruco_size, use_generic=True,
            aruco_dict=aruco_dict, params=params)
        arucos = select_aruco_poses(arucos, PoseSelectors.Z_axis_up)
        arucos = select_aruco_markers(arucos, lambda id: id >= 4)
        draw_aruco(image, arucos, False, False, K, D)

    if save_folder:
        save_callback = StreamCallbacks.get_save_by_key_callback(save_folder)
    else:
        save_callback = lambda: None
    stream(camera, [save_callback, detect_and_draw_aruco_on_boxes],
        "stream aruco detected on boxes")
