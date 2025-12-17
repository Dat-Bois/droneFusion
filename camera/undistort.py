import cv2
import numpy as np
from pathlib import Path
from typing import Tuple

class Undistort:
    def __init__(self, intrinsics : Path | str | np.ndarray, distortion : Path | str | np.ndarray, size : Tuple[int, int] = (1920, 1080)):
    
        if isinstance(intrinsics, (Path, str)):
            intrinsics = np.loadtxt(intrinsics)
        if isinstance(distortion, (Path, str)):
            distortion = np.loadtxt(distortion)

        self.intrinsics = intrinsics
        self.dist = distortion
        self.size = size

        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(intrinsics, distortion, size, 1, size)
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(intrinsics, distortion, None, self.new_camera_matrix, size, 5)

    def undistort_only(self, frame, *, cuda = False):
        if cuda:
            cuMapX = cv2.cuda.GpuMat(self.mapx)
            cuMapY = cv2.cuda.GpuMat(self.mapy)
            cuFrame = cv2.cuda.GpuMat(frame)
            undistorted_img = cv2.cuda.remap(cuFrame, cuMapX, cuMapY, cv2.INTER_LINEAR)
            return undistorted_img.download()
        return cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)

    def crop_roi(self, frame):
        x, y, w, h = self.roi
        return frame[y:y+h, x:x+w]
    
    def get_roi_dimensions(self):
        x, y, w, h = self.roi
        return ((x+w)-x, (y+h)-y)

    def undistort(self, frame, *, with_cuda = False):
        undistorted_img = self.undistort_only(frame, cuda = with_cuda)
        return self.crop_roi(undistorted_img)