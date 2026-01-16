import cv2
import time
import pickle
import pathlib
import threading
import queue
from camera import Camera, Image
from radar import MmWaveRadar

'''
Initializes and records the cameras and radar data to files for later processing.

At 20hz every frame is saved to disk titled camera_name_timestamp
At the same rate, radar data is saved to radar_timestamp.pkl
'''

#---Setable parameters---

SAVE_PATH = "/media/eesh/RECORD"
save_radar = True
FPS_LIMIT = 20
show_camera = "optical_wide"

#-------------

force_quit = False
running = False
frame_queue = queue.Queue()

def write_thread():
    while running or (not frame_queue.empty() and not force_quit):
        time.sleep(0.001)
        try:
            item = frame_queue.get(timeout=1)
            if item is None:
                continue
            filename, data = item
            if data is None:
                continue
            if isinstance(data, Image):
                cv2.imwrite(filename, data.frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            else:
                with open(filename, 'wb') as f:
                    pickle.dump(data, f)
        except queue.Empty:
            continue

now = time.localtime()
formatted_time = time.strftime("%m_%d_%H_%M", now)
# Create folder for data
SAVE_PATH = SAVE_PATH + f"/test_{formatted_time}/"
pathlib.Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)

print(f"Saving data to: {SAVE_PATH}")

# initialize cameras (specify bus addresses)
cameras : list[Camera] = []
opt_camera = Camera(bus_addr=[1,4], camera_type='optical_wide', format='BGR', resolution=(1920,1080), fps=30, camera_name='optical_wide')
cameras.append(opt_camera)

# optn_camera = Camera(bus_addr=[1,4], camera_type='optical_wide', format='BGR', resolution=(1920,1080), fps=30, camera_name='optical_narrow')
# cameras.append(optn_camera)

therm_camera = Camera(bus_addr=[1,14], camera_type='optical_wide', format='GRAY8', resolution=(640,480), fps=30, camera_name='thermal_wide')
cameras.append(therm_camera)

# thermn_camera = Camera(bus_addr=[1,7], camera_type='optical_wide', format='GRAY8', resolution=(640,480), fps=30, camera_name='thermal_narrow')
# cameras.append(thermn_camera)

# Start write thread
running = True
wt = threading.Thread(target=write_thread, daemon=True)
wt.start()

available_cams = []
print("Starting cameras and radar...")
# Start cameras
for cam in cameras:
    print(f"Starting camera: {cam.camera_name}")
    cam.start()
    available_cams.append(cam.camera_name)

if show_camera:
    if show_camera in available_cams:
        cv2.namedWindow(show_camera, cv2.WINDOW_NORMAL) 
        cv2.resizeWindow(show_camera, 1280, 720)
    else:
        print(f"Warning: Requested camera to show '{show_camera}' is not available.")
        show_camera = False

time.sleep(10)
# Connect and start radar
if save_radar:
    radar = MmWaveRadar(cfg_path="/home/eesh/droneFusion/radar/configs/1843_short_range.cfg", already_started=False, verbose=True)
    radar.connect()
    radar.configure()
else:
    print("Skipping radar initialization.")
time.sleep(2)

# Adding configuration info to save folder
with open(SAVE_PATH + "config.txt", 'w') as f:
    f.write(f"Using Cameras: {available_cams}\n")
    f.write(f"Using Radar: {save_radar}\n")
    f.write(f"Radar Config: {radar.cfg_path if save_radar else 'N/A'}\n")
    f.write(f"FPS Limit: {FPS_LIMIT}\n")
    f.write(f"Show Camera: {show_camera}\n")
    f.write(f"Save Path: {SAVE_PATH}\n")

print("Recording data... (Press Ctrl+C to stop)")

stop = False
while True:
    try:
        for cam in cameras:
            if not cam.stream:
                stop = True
                break
        if stop:
            break
        start_time = time.time()
        for cam in cameras:
            frame : Image = cam.get_latest_frame(undistort=False)
            if frame is not None:
                timestamp = int(time.time() * 1e6) # microseconds
                filename = SAVE_PATH + f"{cam.camera_name}_{timestamp}.jpg"
                frame.frame = cv2.rotate(frame.frame, cv2.ROTATE_180) # Rotate if needed
                frame_queue.put((filename, frame))
                if show_camera and cam.camera_name == show_camera:
                    cv2.imshow(show_camera, frame.frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        if save_radar:
            data = radar.get_frame()
            if data is not None:
                timestamp = int(time.time() * 1e6) # microseconds
                filename = SAVE_PATH + f"radar_{timestamp}.pkl"
                frame_queue.put((filename, data))
        elapsed = time.time() - start_time
        sleep_time = max(0, (1.0 / FPS_LIMIT) - elapsed)
        time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        break

for cam in cameras:
    cam.stop()
if save_radar:
    radar.close()
cv2.destroyAllWindows()
running = False
print("Waiting for write queue to finish... (Ctrl+C again to force quit)")
try:
    while not frame_queue.empty():
        print(f"Frames remaining in queue: {frame_queue.qsize()}")
        time.sleep(1)
except KeyboardInterrupt:
    print("Force quitting...")
    force_quit = True
wt.join()
print("Done.")
