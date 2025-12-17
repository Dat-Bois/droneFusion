import os
import numpy as np
import cv2
from pathlib import Path

CURRENT_FILE_PATH = Path(__file__).parent.absolute()
IMG_FILE_PATH = CURRENT_FILE_PATH / "opt_wide"

"""
Creates the camera projection model based on camera instrinics
  
    Outputs:
    instrinics (np.ndarray): matrix of shape (3,4) representing the camera instrinics matrix
              [[f_x, 0, o_y],
               [0, f_y, o_y],
               [0,   0,  1 ]]
  
    The values fx and fy are the pixel focal length, and are identical for square pixels. 
    The values ox and oy are the offsets of the principal point from the top-left corner of the image frame. 
    All values are expressed in pixels.
  
"""

def calibrate_camera(*, img_directory = IMG_FILE_PATH, save_file = False):

    # Initialize cv2 window
    cv2.namedWindow("Checkerboard", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("Checkerboard", 640, 480)

    # Load all calibration images from the img_directory
    img_directory_ls = os.listdir(img_directory)
    img_list = []
    for f_name in img_directory_ls:
        if not f_name.endswith('.jpg'):
            continue
        img = cv2.imread( os.path.join(img_directory, f_name))
        img_list.append(img)
        print(f'loaded image {f_name}')

    # Define the chessboard dimensions (MAKE SURE THESE ARE ACCURATE)
    chess_board_col_points, chess_board_row_points, real_coord_axis = 6, 9, 3

    # Define termination criteria
    critera = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros(( chess_board_col_points * chess_board_row_points, real_coord_axis), np.float32)
    objp[ :, :2] = np.mgrid[ 0: chess_board_row_points, 0: chess_board_col_points].T.reshape( -1, 2) 

    # objpoints is storing points in the real world coordinate system while imgpoints is storing points in the 2d image coordinate system
    objpoints = []
    imgpoints = []

    for i, camera_photo in enumerate(img_list):
        print(f'Processing image_{i+1}')
        gray = cv2.cvtColor(camera_photo, cv2.COLOR_BGR2GRAY)
        cv2.imshow(f'Checkerboard', gray)
        cv2.waitKey(500)
        ret, corners = cv2.findChessboardCorners( gray, (chess_board_row_points, chess_board_col_points), None)
        print(ret)
        # corner of chessboard has been found, adding point to the two lists
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix( gray, corners, ( 11, 11), (-1, -1), critera)
            imgpoints.append(corners2)

            cv2.drawChessboardCorners(gray, (chess_board_row_points, chess_board_col_points), corners2, ret)
            cv2.imshow(f'Checkerboard', gray)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # set up code to run this function
    ret, intrinsics_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], intrinsics_matrix, dist)
        error = cv2.norm( imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    print(f"total reprojection error: { mean_error / len(objpoints)}")

    if ret <= 0:
        print('UNSUCCESSFUL CALIBRATION, NOT SETTING INSTRINICS')
        return None
    else:
        print("SUCCESSFUL CALIBRATION SETTING INSTRINICS")
        np.savetxt(img_directory / 'camera_intrinsic_matrix.txt', intrinsics_matrix)
        np.savetxt(img_directory / 'camera_distortion_matrix.txt', dist)
        print(f'INSTRINICS MATRIX {intrinsics_matrix}')
        return intrinsics_matrix
    
if __name__ == "__main__":
    calibrate_camera(save_file=True)