import os
import numpy as np
import cv2
from pathlib import Path

CURRENT_FILE_PATH = Path(__file__).parent.absolute()
IMG_FILE_PATH = CURRENT_FILE_PATH / "opt_wide"

# --- CHECKERBOARD SETTINGS ---
CHESS_BOARD_DIMS = (9, 6) # (Rows, Columns) of internal corners

# --- CHARUCO SETTINGS ---
CHARUCO_SQUARES_X = 8
CHARUCO_SQUARES_Y = 11
CHARUCO_SQUARE_LENGTH = 500 
CHARUCO_MARKER_LENGTH = 300 
CHARUCO_DICT_TYPE = cv2.aruco.DICT_6X6_250

def get_charuco_board():
    """Generates the ChArUco board object based on constants."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(CHARUCO_DICT_TYPE)
    board = cv2.aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
        CHARUCO_SQUARE_LENGTH,
        CHARUCO_MARKER_LENGTH,
        aruco_dict
    )
    return board, aruco_dict

def detect_calibration_type(img):
    """
    Detects if the image contains ArUco markers to decide the calibration mode.
    Returns: 'charuco' or 'checkerboard'
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(CHARUCO_DICT_TYPE)
    parameters = cv2.aruco.DetectorParameters()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    if ids is not None and len(ids) > 0:
        return 'charuco'
    
    return 'checkerboard'

def calibrate_camera(*, img_directory=IMG_FILE_PATH, save_file=False):
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("Calibration", 640, 480)
    img_directory_ls = os.listdir(img_directory)
    img_list = []
    print(f"Loading images from {img_directory}...")
    for f_name in sorted(img_directory_ls):
        if not f_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        img_path = os.path.join(img_directory, f_name)
        img = cv2.imread(img_path)
        if img is not None:
            img_list.append(img)
            # print(f'Loaded {f_name}')
    
    if not img_list:
        print("No images found!")
        return None

    mode = detect_calibration_type(img_list[0])
    print(f"--- DETECTED MODE: {mode.upper()} ---")
    
    if mode == 'checkerboard':
        ret, mtx, dist, rvecs, tvecs = _process_checkerboard(img_list, CHESS_BOARD_DIMS)
    else:
        ret, mtx, dist, rvecs, tvecs = _process_charuco(img_list)

    cv2.destroyAllWindows()

    if ret is None or (isinstance(ret, bool) and not ret):
        print('UNSUCCESSFUL CALIBRATION, NOT SETTING INTRINSICS')
        return None
    
    print("\nSUCCESSFUL CALIBRATION")
    print(f"RMS Re-projection Error: {ret}")

    if save_file:
        print("Saving matrix files...")
        np.savetxt(img_directory / 'camera_intrinsic_matrix.txt', mtx)
        np.savetxt(img_directory / 'camera_distortion_matrix.txt', dist)
        print(f'INTRINSICS MATRIX:\n{mtx}')
    
    return mtx

def _process_checkerboard(img_list, board_dims):
    """Handles standard chessboard calibration logic."""
    
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((board_dims[0] * board_dims[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_dims[0], 0:board_dims[1]].T.reshape(-1, 2)

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    for i, img in enumerate(img_list):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, board_dims, None)

        if ret == True:
            print(f"Image {i+1}: Corners Found")
            objpoints.append(objp)
            
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            cv2.drawChessboardCorners(img, board_dims, corners2, ret)
            cv2.imshow('Calibration', img)
            cv2.waitKey(200) # Short wait to visualize
        else:
            print(f"Image {i+1}: Corners NOT Found")

    if len(objpoints) > 0:
        return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return None, None, None, None, None

def _process_charuco(img_list):
    """Handles ChArUco calibration logic."""
    
    board, aruco_dict = get_charuco_board()
    params = cv2.aruco.DetectorParameters()

    all_charuco_corners = []
    all_charuco_ids = []

    for i, img in enumerate(img_list):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
        if len(corners) > 0:
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board
            )
            if retval > 0 and charuco_corners is not None and len(charuco_corners) > 4:
                print(f"Image {i+1}: {len(charuco_corners)} ChArUco corners found")
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
                cv2.aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)
                cv2.imshow('Calibration', img)
                cv2.waitKey(200)
            else:
                print(f"Image {i+1}: Markers found, but ChArUco interpolation failed")
        else:
            print(f"Image {i+1}: No ArUco markers found")

    if len(all_charuco_corners) > 0:
        image_size = img_list[0].shape[:2][::-1]
        
        # calibrateCameraCharuco returns (repError, mtx, dist, rvecs, tvecs)
        # Note: We pass None for cameraMatrix and distCoeffs to let OpenCV estimate them
        return cv2.aruco.calibrateCameraCharuco(
            all_charuco_corners, 
            all_charuco_ids, 
            board, 
            image_size, 
            None, 
            None
        )
    return None, None, None, None, None

if __name__ == "__main__":
    calibrate_camera(save_file=True)