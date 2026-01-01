import os
os.environ['YOLO_VERBOSE'] = 'False'

import cv2
import time
import numpy as np
from camera import Camera, Image, Results
from fusion import Tracker, Measurement, PoseStamped

import traceback

def result_to_measurements(camera : Camera, result : Results) -> list[Measurement]:
    '''
    Converts a single Results object from a camera into a Measurement object for the tracker.
    Filters the results based on confidence and class (people only > 70%).
    Uses camera interinsics to and assumed distance to estimate 3D position.
    '''
    measurements = []
    confidence = 0.6
    intrinsics, opt_matrix = camera.get_camera_matrices()
    names = result.names
    for box in result.boxes:
        conf = box.conf.item()
        cls_id = box.cls.item()
        if conf < confidence or names[int(cls_id)] != 'person':
            continue
        x1, y1, x2, y2 = box.xyxy.cpu().numpy().flatten()
        timestamp = time.time()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        # Assume a fixed distance of 2 meters in front of the camera
        Z = 2.0
        k_inv = np.linalg.inv(opt_matrix)
        uv1 = np.array([cx, cy, 1.0])
        xyz_camera = k_inv @ uv1
        scale = Z / xyz_camera[2]
        X = xyz_camera[0] * scale
        Y = xyz_camera[1] * scale
        position = np.array([X, Y, Z])
        measurement = Measurement(timestamp=timestamp, pose=position, R=np.eye(3)*0.2)
        measurements.append(measurement)
    return measurements

def T3D_to_2D(camera : Camera, position_3D : np.ndarray) -> np.ndarray:
    '''
    Projects a 3D position in camera frame to 2D pixel coordinates using camera intrinsics.
    '''
    intrinsics, opt_matrix = camera.get_camera_matrices()
    x, y, z = position_3D
    uv1 = opt_matrix @ np.array([x, y, z])
    u = uv1[0] / uv1[2]
    v = uv1[1] / uv1[2]
    return np.array([u, v])

tracker = Tracker()

opt_camera = Camera(bus_addr=[1,5], camera_type='optical_wide', format='BGR', resolution=(1920,1080), fps=30, camera_name='optical_wide')
opt_camera.switch_model("yolo11n.pt")
opt_camera.start()
time.sleep(2)

fps = 0.0
while opt_camera.stream:
    try:
        pre_time = time.time()
        frame_opt : Image = opt_camera.get_latest_frame()
        if frame_opt is not None:
            frame_opt = frame_opt.frame
            results = opt_camera.get_latest_model_results()
            if results is not None:
                for result in results:
                    measurements = result_to_measurements(opt_camera, result)
                    for measurement in measurements:
                        tracker.update(measurement)
            tracks = tracker.get_latest_tracks(timestamp=time.time())
            for pose_stamped in tracks:
                position_3D = pose_stamped.state[:3].flatten()
                pixel_coords = T3D_to_2D(opt_camera, position_3D)
                u, v = int(pixel_coords[0]), int(pixel_coords[1])
                cv2.circle(frame_opt, (u, v), 10, (0, 255, 0), -1)
                cv2.putText(frame_opt, f"ID: {pose_stamped.track_id}", (u + 15, v), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # cv2.putText(frame_opt, f"R: {pose_stamped.dist:.2f}m", (u + 15, v + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # cv2.putText(frame_opt, f"S: {pose_stamped.mahalonobis:.2f}", (u + 15, v + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            post_opt_time = time.time() - pre_time

        # opt_frame = opt_camera.draw_model_results(frame_opt, confidence=0.6)

        # Write FPS on the top left corner
        new_fps = 1 / post_opt_time
        if new_fps < 60:
            fps = new_fps
        cv2.putText(frame_opt, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Optical", frame_opt)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    except Exception as e:
        print(e)
        # traceback.print_exc()
        break

tracker.stop_tracker()
opt_camera.stop()
cv2.destroyAllWindows()
