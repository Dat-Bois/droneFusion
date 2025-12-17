import os
os.environ['YOLO_VERBOSE'] = 'False'

import cv2
import time
from camera import Camera, Image

# Create a camera object
camera = Camera(bus_addr=[1,10], camera_type='port')
camera.switch_model("yolo11n.pt")

camera.start()

cv2.namedWindow("Yolo", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("Yolo", 1280, 720)
pre_results = None
fps = 0
while camera.stream:

    pre_time = time.time()

    frame : Image = camera.get_latest_frame()
    if frame is None:
        continue
 
    frame = frame.frame
    frame = camera.draw_model_results(frame, confidence=0.6)
    post_time = time.time() - pre_time

    # Write FPS on the top left corner
    new_frame_time = time.time()
    new_fps = 1 / (new_frame_time - pre_time)
    if new_fps < 60:
        fps = new_fps
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if frame is not None:
        cv2.imshow("Yolo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.stop()
cv2.destroyAllWindows()