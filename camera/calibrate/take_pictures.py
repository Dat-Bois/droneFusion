import cv2
import os
from pathlib import Path
import time

#optical
cap = cv2.VideoCapture("v4l2src device=/dev/video6 ! image/jpeg, width=1920, height=1080,framerate=30/1 ! jpegdec ! videoconvert ! video/x-raw, format=BGR ! appsink ")

#thermal
# cap = cv2.VideoCapture("v4l2src device=/dev/video0 ! image/jpeg, width=640, height=480,framerate=30/1 ! jpegdec ! videoconvert ! video/x-raw, format=GRAY8 ! appsink ")
'''
1,4 narrow opt
1,6 wide thermal
1,7 narrow thermal
1,8 wide opt
'''

CURRENT_FILE_PATH = Path(__file__).parent.absolute()
IMG_FILE_PATH = CURRENT_FILE_PATH / "opt_wide"

if not IMG_FILE_PATH.exists():
    os.mkdir(IMG_FILE_PATH)

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("Camera", 640, 480)

i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: failed to capture image")
        continue

    cv2.imshow("Camera", frame)
    
    k = cv2.waitKey(1) & 0xFF

    if k == ord('q'):
        print("Exiting...")
        break

    elif k == ord('s'):
        i += 1
        file_name = f"image_{i}_{time.time()}.jpg"
        print(f'saving image: {file_name}')
        cv2.imwrite(str(IMG_FILE_PATH / file_name), frame)

    if i == 20:
        print("Max number of images saved")
        break

cap.release()
cv2.destroyAllWindows()