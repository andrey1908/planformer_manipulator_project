import numpy as np
import cv2
import os
import os.path as osp
from segmentation import segment_red_boxes_hsv, segment_blue_boxes_hsv, \
    segment_goal_hsv, segment_stop_line_hsv


def prepare_prompts(input_folder, output_folder):
    os.makedirs(osp.join(output_folder, "segm"), exist_ok=False)
    os.makedirs(osp.join(output_folder, "img"), exist_ok=False)
    for camera in ("front", "top"):
        image_file = osp.join(input_folder, f"{camera}.jpg")
        image = cv2.imread(image_file)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
        # show(hsv)
        for prompt in ("red_box", "blue_box"):
            if prompt == "red_box":
                segmentation, _, _ = segment_red_boxes_hsv(hsv)
            elif prompt == "blue_box":
                segmentation, _, _ = segment_blue_boxes_hsv(hsv)
            segmentation *= 255
            # show(segmentation)

            if camera == "front" and prompt == "red_box":
                x = 746
                y = 469
            if camera == "front" and prompt == "blue_box":
                x = 535
                y = 469
            if camera == "top" and prompt == "red_box":
                x = 561
                y = 525
            if camera == "top" and prompt == "blue_box":
                x = 655
                y = 512
            w = 256
            h = 128
            half_w = int(w / 2)
            half_h = int(h / 2)
            segm = segmentation[y - half_h:y + half_h, x - half_w: x + half_w]
            # show(segm)
            if camera == "top":
                segm_to_save = 255 - cv2.rotate(segm, cv2.ROTATE_180)
            else:
                segm_to_save = 255 - segm
            cv2.imwrite(osp.join(output_folder, f"segm/{prompt}_{camera}.png"), segm_to_save)
            crop = image[y - half_h:y + half_h, x - half_w: x + half_w]
            img = np.ones(crop.shape, dtype=np.uint8) * 255
            mask = segm == 255
            img[mask] = crop[mask]
            if camera == "top":
                img_to_save = cv2.rotate(img, cv2.ROTATE_180)
            else:
                img_to_save = img
            cv2.imwrite(osp.join(output_folder, f"img/{prompt}_{camera}.png"), img_to_save)

    for camera in ("front", "top"):
        image_file = osp.join(input_folder, f"{camera}.jpg")
        image = cv2.imread(image_file)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
        # show(hsv)
        for prompt in ("stop_line",):
            if prompt == "stop_line":
                segmentation, _, _ = segment_stop_line_hsv(hsv)
            segmentation *= 255
            # show(segmentation)

            if camera == "front" and prompt == "stop_line":
                x = 643
                y = 345
            if camera == "top" and prompt == "stop_line":
                x = 631
                y = 642
            w_scale = 2
            w = 256 * w_scale
            h = 128
            half_w = int(w / 2)
            half_h = int(h / 2)
            segm = segmentation[y - half_h:y + half_h, x - half_w: x + half_w]
            # show(segm)
            segm_embedded = np.zeros((h, int(w / w_scale)), dtype=np.uint8)
            segm_embedded[int(half_h / 2): int(half_h / 2) + half_h, :] = \
                cv2.resize(segm, (int(w / w_scale), int(h / w_scale)), interpolation=cv2.INTER_NEAREST)
            if camera == "top":
                segm_to_save = 255 - cv2.rotate(segm_embedded, cv2.ROTATE_180)
            else:
                segm_to_save = 255 - segm_embedded
            cv2.imwrite(osp.join(output_folder, f"segm/{prompt}_{camera}.png"), segm_to_save)
            crop = image[y - half_h:y + half_h, x - half_w: x + half_w]
            img = np.ones(crop.shape, dtype=np.uint8) * 255
            mask = segm == 255
            img[mask] = crop[mask]
            img_embedded = np.ones((h, int(w / w_scale), 3), dtype=np.uint8) * 255
            img_embedded[int(half_h / 2): int(half_h / 2) + half_h, :] = \
                cv2.resize(img, (int(w / w_scale), int(h / w_scale)), interpolation=cv2.INTER_NEAREST)
            if camera == "top":
                img_to_save = cv2.rotate(img_embedded, cv2.ROTATE_180)
            else:
                img_to_save = img_embedded
            cv2.imwrite(osp.join(output_folder, f"img/{prompt}_{camera}.png"), img_to_save)

    for camera in ("front", "top"):
        image_file = osp.join(input_folder, f"{camera}_goal.jpg")
        image = cv2.imread(image_file)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
        for prompt in ("goal",):
            if prompt == "goal":
                segmentation, _, _ = segment_goal_hsv(hsv)
            segmentation *= 255
            # show(segmentation)

            if camera == "front" and prompt == "goal":
                x = 638
                y = 370
            if camera == "top" and prompt == "goal":
                x = 648
                y = 587
            scale = 3
            w = 256 * scale
            h = 128 * scale
            half_w = int(w / 2)
            half_h = int(h / 2)
            segm = segmentation[y - half_h:y + half_h, x - half_w: x + half_w]
            segm_resized = cv2.resize(segm, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_NEAREST)
            # show(segm)
            if camera == "top":
                segm_to_save = 255 - cv2.rotate(segm_resized, cv2.ROTATE_180)
            else:
                segm_to_save = 255 - segm_resized
            cv2.imwrite(osp.join(output_folder, f"segm/{prompt}_{camera}.png"), segm_to_save)
            crop = image[y - half_h:y + half_h, x - half_w: x + half_w]
            img = np.ones(crop.shape, dtype=np.uint8) * 255
            mask = segm == 255
            img[mask] = crop[mask]
            img_resized = cv2.resize(img, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_NEAREST)
            if camera == "top":
                img_to_save = cv2.rotate(img_resized, cv2.ROTATE_180)
            else:
                img_to_save = img_resized
            cv2.imwrite(osp.join(output_folder, f"img/{prompt}_{camera}.png"), img_to_save)