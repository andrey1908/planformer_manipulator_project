import cv2
import numpy as np
from aruco.aruco import detect_aruco, draw_aruco, select_aruco_poses, get_aruco_corners_3d, PoseSelectors
from estimate_plane_frame.estimate_plane_frame import estimate_plane_frame
from realsense_camera.realsense_camera import RealsenseCamera


def show(image):
    cv2.imshow('', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def stream_table_frame(K, D, aruco_size, aruco_dict, params, save_vid=None, debug=False):
    camera = RealsenseCamera()
    camera.start()
    i = 0
    had_detection = False
    if save_vid is not None:
        vid = cv2.VideoWriter(save_vid, cv2.VideoWriter_fourcc(*'mp4v'),
            30, (1280, 720))
    while True:
        image = camera()
        if image is None:
            continue
        arucos = detect_aruco(image, K=K, D=D, aruco_sizes=aruco_size,
            use_generic=False, aruco_dict=aruco_dict, params=params)
        if arucos.n == 4:
            corners_3d = get_aruco_corners_3d(arucos).reshape(16, 3)
            camera2table = estimate_plane_frame(corners_3d)
            rvec, _ = cv2.Rodrigues(camera2table[0:3, 0:3])
            tvec = camera2table[0:3, 3]
            cv2.drawFrameAxes(image, K, D, rvec, tvec, 0.1)
            had_detection = True
        elif had_detection and debug:
            cv2.imwrite(f'realsense/debug/{i:04}.png', image)
            i += 1
        if save_vid is not None:
            vid.write(image)
        cv2.namedWindow('stream table frame', cv2.WINDOW_NORMAL)
        cv2.imshow('stream table frame', image)
        key = cv2.waitKey(1)
        if key != -1:
            cv2.destroyAllWindows()
            camera.stop()
            break

    if save_vid is not None:
        vid.release()
