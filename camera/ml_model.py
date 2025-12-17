import cv2
import time
import numpy as np
from typing import List
from pathlib import Path
from threading import Thread, Lock
from ultralytics import YOLO
from ultralytics.engine.results import Results


class ML_Model:

    def __init__(self, model_path: str | Path, model_type: str = "YOLO", *, half_precision: bool = False):
        self._set_params(model_path, model_type, half_precision=half_precision)

    def switch_model(self, model_path: str | Path, model_type: str = "YOLO", *, half_precision: bool = False):
        self._reset_params()
        self._set_params(model_path, model_type, half_precision=half_precision)
        print(f'Model switched successfully to: {model_type} \n \
              With weights: {model_path}')

    def _set_params(self, model_path: str | Path, model_type: str, *, half_precision):
        if not isinstance(model_path, Path):
            model_path = Path(model_path)
        self.model_path = model_path
        self.model_type = model_type.lower()
        self.half_precision = half_precision
        self.model = None
        self._async_init_model()

    def _async_init_model(self):
        init_thread = Thread(target=self._init_model)
        init_thread.start()
        init_thread.join()

    def _init_model(self):
        if self.model_type == "yolo":
            if self.model_path.suffix != ".pt":
                raise ValueError("Model path must be a .pt file for YOLO models")
            self.model = YOLO(self.model_path, task="detect", verbose=False)
        elif self.model_type == "tensorrt":
            if self.model_path.suffix != ".engine":
                raise ValueError("Model path must be a .engine file for TensorRT models. Please use \
                                    the create_tensorrt_model() to generate a TensorRT model from the .pt file.")
            self.model = YOLO(self.model_path, task="detect", verbose=False)
        else:
            raise ValueError("Invalid model type. Please choose between YOLO and TensorRT")
        print(f"Model initialized successfully: {self.model_type}")

    def _reset_params(self):
        del self.model
        self.model = None

    def predict(self, frame: np.ndarray) -> list[Results]:
        if self.model is None or frame is None:
            return []
        results: list[Results] = self.model(frame, half=self.half_precision)
        return results

    def create_tensorrt_model(self, model_path: str | Path = None):
        if model_path == None:
            model_path = self.model_path
        print("Creating TensorRT model...please be patient")
        model = YOLO(model_path)
        model.export(format="engine", half=self.half_precision)
        print("Model created successfully")

if __name__ == "__main__":
    model = ML_Model("models/yolov8n.pt")
    model.create_tensorrt_model(half_precision=True)