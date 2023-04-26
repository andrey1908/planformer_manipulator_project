import numpy as np
import cv2
import os
import os.path as osp
from aruco import detect_aruco, select_aruco_poses, PoseSelectors
from segmentation import segment_and_draw_boxes_by_aruco
from aruco_detection_configs import aruco_dict, aruco_detection_params


def create_folders(out_folder):
    os.makedirs(osp.join(out_folder, "accepted/aruco"), exist_ok=False)
    os.makedirs(osp.join(out_folder, "accepted/box"), exist_ok=False)
    os.makedirs(osp.join(out_folder, "rejected/aruco"), exist_ok=False)
    os.makedirs(osp.join(out_folder, "rejected/box"), exist_ok=False)


def check_images(aruco_image, box_image, K, D, aruco_size, view):
    arucos = detect_aruco(aruco_image, K=K, D=D, aruco_sizes=aruco_size, use_generic=True,
        aruco_dict=aruco_dict, params=aruco_detection_params)
    if view == "top":
        arucos = select_aruco_poses(arucos, PoseSelectors.best)
    elif view == "front":
        arucos = select_aruco_poses(arucos, PoseSelectors.Z_axis_up)
    print(f"Detected {arucos.n} aruco{'s' if arucos.n != 1 else ''}")

    draw = box_image.copy()
    segment_and_draw_boxes_by_aruco(draw, arucos, K, D)

    while True:
        print("Accept image? [y/n]")
        cv2.imshow(view, draw)
        ret = cv2.waitKey(0)
        if ret == ord('y'):
            print("Accepted!")
            return True
        elif ret == ord('n'):
            print("Rejected.")
            return False
        else:
            print("Unknown answer")


def stream_camera(camera, view):
    while True:
        frame = camera()
        cv2.imshow(view, frame)
        ret = cv2.waitKey(1)
        if ret == 'q':
            exit(0)
        elif ret != -1:
            return frame


def collect_dataset(camera, view, K, D, aruco_size, out_folder):
    assert view in ("top", "front")
    create_folders(out_folder)
    cv2.namedWindow(view, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(view, 600, 300)
    image_id = 0
    while True:
        print("Take aruco image...")
        aruco_image = stream_camera(camera, view)
        print("Aruco image is taken")

        print("Take box image...")
        box_image = stream_camera(camera, view)
        print("Box image is taken")

        accepted = check_images(aruco_image, box_image, K, D, aruco_size, view)
        if accepted:
            cv2.imwrite(osp.join(out_folder, f"accepted/aruco/{image_id:04}.png"), aruco_image)
            cv2.imwrite(osp.join(out_folder, f"accepted/box/{image_id:04}.png"), box_image)
        else:
            cv2.imwrite(osp.join(out_folder, f"rejected/aruco/{image_id:04}.png"), aruco_image)
            cv2.imwrite(osp.join(out_folder, f"rejected/box/{image_id:04}.png"), box_image)

        image_id += 1
