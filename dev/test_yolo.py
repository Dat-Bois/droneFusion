import os
os.environ['YOLO_VERBOSE'] = 'False'

import cv2
import time
from camera import Camera, Image

'''
1,4 optical_narrow
1,6 thermal_wide
1,7 thermal_narrow
1,8 optical_wide
'''

# Create a camera object
opt_camera = Camera(bus_addr=[1,4], camera_type='optical_wide', format='BGR', resolution=(1920,1080), fps=30, camera_name='optical_wide')
# therm_camera = Camera(bus_addr=[1,7], camera_type='optical_wide', format='GRAY8', resolution=(640,480), fps=30, camera_name='thermal_wide')

opt_camera.switch_model("yolo11n.pt")
# therm_camera.switch_model("yolo11n.pt")

opt_camera.start()
# therm_camera.start()
time.sleep(2)

cv2.namedWindow("Optical", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("Optical", 1280, 720)

# cv2.namedWindow("Thermal", cv2.WINDOW_NORMAL) 
# cv2.resizeWindow("Thermal", 1280, 720)
frame_therm = None
while opt_camera.stream: #and therm_camera.stream:

    opt_camera.fps_start()

    frame_opt : Image = opt_camera.get_latest_frame(undistort=False)
    # frame_therm : Image = therm_camera.get_latest_frame(undistort=False)
    if frame_opt is not None:
        frame_opt = frame_opt.frame
        frame_opt = opt_camera.draw_model_results(frame_opt, confidence=0.6)
        opt_fps, _ = opt_camera.fps_end()
    if frame_therm is not None:
        frame_therm = frame_therm.frame
        # frame_therm = therm_camera.draw_model_results(frame_therm, confidence=0.6)
        # therm_fps, _ = therm_camera.fps_end()

    if frame_opt is not None:
        # Write FPS on the top left corner
        cv2.putText(frame_opt, f"FPS: {opt_fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Optical", frame_opt)

    if frame_therm is not None:
        # Write FPS on the top left corner
        cv2.putText(frame_therm, f"FPS: {therm_fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Thermal", frame_therm)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

opt_camera.stop()
# therm_camera.stop()
cv2.destroyAllWindows()