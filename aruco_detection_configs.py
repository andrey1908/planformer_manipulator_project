import cv2


def get_aruco_dict():
    dict_4x4 = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
    return dict_4x4


def get_aruco_detection_params():
    params = cv2.aruco.DetectorParameters_create()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    params.perspectiveRemovePixelPerCell = 8
    params.perspectiveRemoveIgnoredMarginPerCell = 0.26
    return params


aruco_dict = get_aruco_dict()
aruco_detection_params = get_aruco_detection_params()