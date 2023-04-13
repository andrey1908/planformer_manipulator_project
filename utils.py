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
        table_detected = camera2table is not None
        if table_detected != calibrate_and_draw_table_frame.table_detected:
            calibrate_and_draw_table_frame.table_detected = table_detected
            if table_detected:
                print("Table detected")
            else:
                print("Cannot detect table")
                return
        rvec, _ = cv2.Rodrigues(camera2table[0:3, 0:3])
        tvec = camera2table[0:3, 3]
        cv2.drawFrameAxes(image, K, D, rvec, tvec, 0.1)
    calibrate_and_draw_table_frame.table_detected = None

    if save_folder:
        save_callback = StreamCallbacks.get_save_by_key_callback(save_folder)
    else:
        save_callback = lambda: None
    stream(camera, [save_callback, calibrate_and_draw_table_frame], "stream table frame")


def stream_segmented_boxes(camera, save_folder=None):
    def segment_and_draw_boxes(image, key):
        mask, (num_red, num_blue) = segment_boxes_by_color(image)
        if num_red != segment_and_draw_boxes.num_red or \
                num_blue != segment_and_draw_boxes.num_blue:
            print(f"Segmented {num_red} red, {num_blue} blue boxes")
            segment_and_draw_boxes.num_red = num_red
            segment_and_draw_boxes.num_blue = num_blue
        polygons, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.polylines(image, polygons, True, (255, 255, 255), thickness=1)
        overlay = image.copy()
        for polygon in polygons:
            cv2.fillPoly(overlay, [polygon], (255, 255, 255))
        cv2.addWeighted(image, 0.7, overlay, 0.3, 0, dst=image)
    segment_and_draw_boxes.num_red = -1
    segment_and_draw_boxes.num_blue = -1

    if save_folder:
        save_callback = StreamCallbacks.get_save_by_key_callback(save_folder)
    else:
        save_callback = lambda: None
    stream(camera, [save_callback, segment_and_draw_boxes], "stream segmented boxes")


def stream_aruco_detected_on_boxes(camera, K, D, aruco_size, aruco_dict, params,
        save_folder=None):
    def detect_and_draw_aruco_on_boxes(image, key):
        arucos = detect_aruco(image, K=K, D=D, aruco_sizes=aruco_size, use_generic=True,
            subtract=100, aruco_dict=aruco_dict, params=params)
        arucos = select_aruco_poses(arucos, PoseSelectors.Z_axis_up)
        arucos = select_aruco_markers(arucos, lambda id: id >= 4)
        if arucos.n != detect_and_draw_aruco_on_boxes.number_of_boxes:
            print(f"Number of boxes: {arucos.n}")
            detect_and_draw_aruco_on_boxes.number_of_boxes = arucos.n
        draw_aruco(image, arucos, False, False, K, D)
    detect_and_draw_aruco_on_boxes.number_of_boxes = -1

    if save_folder:
        save_callback = StreamCallbacks.get_save_by_key_callback(save_folder)
    else:
        save_callback = lambda: None
    stream(camera, [save_callback, detect_and_draw_aruco_on_boxes],
        "stream aruco detected on boxes")
