import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
from threading import Thread, Lock
from camera.undistort import Undistort
from camera.ml_model import ML_Model, Results
from camera.find_camera import FindCamera
import time

'''
This is the main camera class that will be used to interact with the camera.
It will handle the camera undistortion and getting the latest frame.
The Camera object is imported into your code and used to get the latest frame.
'''

class Image:
    def __init__(self, frame : np.ndarray):
        self.frame = frame
        self.dimensions = (frame.shape[1], frame.shape[0])

class Camera:

    def __init__(self, /, camera_id : int = 0, camera_name : str = "Unknown Camera", *, model : ML_Model = None, 
                 resolution : Tuple[int, int] = (1920, 1080), fps : int = 30, 
                 video_path : str | Path = None, camera_type : str = 'port', 
                 bus_addr : Tuple[int, int] = None):

        self._init_camera_path(camera_id, bus_addr)
        self.video_path = video_path
        self.resolution = resolution
        self.fps = fps
        self.model = model
        self.camera_name = camera_name
        self._load_calibration(camera_type)

        self.stream = False
        self.run_model = False

        self.done_init = False

        self.raw_frame = None
        self.frame = None
        self.results = []
        
        self.camera_lock = Lock()
        self.model_lock = Lock()
        
    def _error(self, message : str):
        print(f"{self.camera_name} Error: {message}")

    def _info(self, message : str):
        print(f"{self.camera_name} Info: {message}")

    def _init_camera_path(self, camera_id : int, bus_addr : Tuple[int, int]):
        if bus_addr is not None:
            fc = FindCamera()
            self.camera_path = fc.find_cam(bus_addr[0], bus_addr[1])
            if self.camera_path is None:
                raise ValueError("Error: Camera not found")
        else:
            self.camera_path = f'/dev/video{camera_id}'
    
    def _load_calibration(self, camera_type : str):
        pre_path = Path(__file__).parent.absolute() / "config"
        if not (pre_path / camera_type).exists():
            camera_type = "port"
        dist_calibration_path = pre_path / Path(f'{camera_type}/camera_distortion_matrix.txt')
        int_calibration_path = pre_path / Path(f'{camera_type}/camera_intrinsic_matrix.txt')
        self.undistort = Undistort(int_calibration_path, dist_calibration_path, self.resolution)

    def load_model_object(self, model_object : ML_Model):
        if not isinstance(model_object, ML_Model):
            self._error("Must be of ML_Model type")
            return
        with self.model_lock:
            self.model = model_object

    def load_model(self, model_path : str | Path, model_type : str = "YOLO", *, half_precision: bool = False):
        if not isinstance(model_path, Path):
            model_path = Path(model_path)
        if not model_path.exists():
            pre_path = Path(__file__).parent.absolute() / "models"
            model_path = pre_path / model_path
            if not model_path.exists():
                self._error("Model path does not exist")
                return
        with self.model_lock:
            self.model = ML_Model(model_path, model_type, half_precision=half_precision)

    def switch_model_object(self, model_object : ML_Model):
        if not isinstance(model_object, ML_Model):
            self._error("Must be of ML_Model type")
            return
        with self.model_lock:
            self.model = model_object

    def switch_model(self, model_path : str | Path, model_type: str = "YOLO", *, half_precision: bool = False):
        if self.model is None:
            self.load_model(model_path, model_type, half_precision=half_precision)
            return
        if not isinstance(model_path, Path):
                model_path = Path(model_path)
        if not model_path.exists():
            pre_path = Path(__file__).parent.absolute() / "models"
            model_path = pre_path / model_path
            if not model_path.exists():
                self._error("Model path does not exist")
                return
        with self.model_lock:
            self.model.switch_model(model_path, model_type, half_precision=half_precision)

    def start(self):
        self.start_stream()
        if self.model:
            self.start_model()

    def stop(self):
        self.stop_model()
        self.stop_stream()

    def start_model(self):
        if not self.stream:
            self._error("Stream not started, cannot start model thread")
            return
        if self.model is None:
            self._error("No model loaded, cannot start model thread")
            return
        if self.run_model:
            self._error("Model thread already running, skipping...")
            return
        self.run_model = True
        self.model_thread = Thread(target=self._model_background_thread)
        self.model_thread.start()
        self._info("Model thread started")

    def stop_model(self):
        self.run_model = False
        try:
            self.model_thread.join()
        except:
            return
        self.results = []
        self._info("Model thread stopped")

    def start_stream(self):
        if self.stream:
            self._error("Stream already started")
            return
        self.warmup()
        self.stream = True
        self.camera_thread = Thread(target=self._camera_background_thread)
        self.camera_thread.start()
        self._info("Stream started")

    def stop_stream(self):
        self.stream = False
        self.done_init = False
        try:
            self.camera_thread.join()
        except:
            return
        self._info("Stream stopped")

    def get_size(self, undistort = True) -> Tuple[int, int]:
        if undistort:
            return self.undistort.get_roi_dimensions()
        return self.resolution
    
    def warmup(self, frame : np.ndarray = None) -> np.ndarray:
        temp_frame = self._warmup_undistort(frame)
        if self.model is not None:
            self.model.predict(temp_frame)

    def _warmup_undistort(self, frame : np.ndarray = None) -> np.ndarray:
        if frame is None:
            # Make a frame with random values
            warmup_frame = np.random.randint(0, 255, (self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        else:
            warmup_frame = frame
        warmup_frame = self.undistort.undistort(warmup_frame, with_cuda=True)
        self._info("Undistortion warmed up...")
        return warmup_frame

    def get_latest_frame(self, *, undistort = True, with_cuda = True) -> Image:
        with self.camera_lock:
            if not self.done_init:
                return None
            # If there is a raw frame use that and then reset it. 
            # Otherwise check if there is a processed frame and use that. If not return None
            if self.raw_frame is not None:
                frame = self.raw_frame
                self.raw_frame = None
            elif self.frame is not None:
                return Image(self.frame)
            else:
                return None
        if undistort:
            frame = self.undistort.undistort(frame, with_cuda = with_cuda)
        self.frame = frame
        return Image(frame)
    
    def get_latest_model_results(self) -> List:
        with self.model_lock:
            return self.results
        
    def draw_model_results(self, frame, results : List[Results] = None, confidence=0.5):
        if results is None:
            results = self.get_latest_model_results()
        for result in results:
            names = result.names
            for box in result.boxes:
                conf = box.conf.item()
                if conf < confidence:
                    continue
                x1, y1, x2, y2 = box.xyxy.cpu().numpy().flatten()
                cls_id = box.cls.item()
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{names[cls_id]}: {conf:.2f}", (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame
    
    def _model_background_thread(self):
        while self.run_model:
            with self.camera_lock:
                frame = self.frame
            if self.stream:
                with self.model_lock:
                    if self.model is not None:
                        self.results = self.model.predict(frame)
            time.sleep(1/(self.fps))

    def _camera_background_thread(self):
        if self.video_path == None:
            capture_flag = f'v4l2src device={self.camera_path} ! image/jpeg, width={self.resolution[0]}, height={self.resolution[1]}, framerate={self.fps}/1 ! jpegdec ! videoconvert ! video/x-raw, format=BGR ! appsink'
            try:
                self.cap = cv2.VideoCapture(capture_flag)
            except:
                self._error("Failed to open camera using GStreamer; Defaulting to OpenCV")
                self.cap = cv2.VideoCapture(self.camera_path)
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        else:
            self.cap = cv2.VideoCapture(self.video_path)
        self.done_init = True
        while self.stream:
            with self.camera_lock:
                ret, self.raw_frame = self.cap.read()
                if not ret:
                    self._error("Failed to capture image")
                    self.stream = False
                    break
            time.sleep(1/(self.fps))

        self.cap.release()
        return
        