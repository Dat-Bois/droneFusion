import numpy as np
from typing import List, Tuple
from queue import PriorityQueue
from threading import Thread, Lock
from fusion import Track, PoseStamped

'''
A simple tracker which takes in 3D measurements, associates them to a respective track via stat and real dist, 
and updates the track state accordingly.

When a track matures, its latest state can be accessed.
When a track hasn't been updated for 5 seconds, it is stored and removed from active tracking.

Every measurement is a 3D numpy array: [x, y, z] (meters) with an associated measurement noise R (3x3 numpy array) and bounding box.
The bounding box is stored with the track and so is its history [width, height] in meters.

General flow (runs in thread):
    - Check active tracks for inactivity and remove inactive ones to stored if they are mature
    - If new measurements are present:
        - Get measurement
        - For each active track, compute stat and real dist based on time with measurement
        - If both dists are below threshold, associate measurement to track
            - Else create new track
        - If measurement works with multiple trakcs, choose the one with lowest stat dist (this can be expanded to MHT later)
        - Update track with measurement

Callable functions:
    - update
    - get_latest_tracks (returns the latest state and bbox for each mature active track) (PoseStamped)
    - get_active_tracks
    - get_stored_tracks
'''

class Measurement:
    def __init__(self, timestamp : float, measurement : np.ndarray, R : np.ndarray, bbox: List[float]):
        self.timestamp = timestamp
        self.measurement = measurement
        self.R = R
        self.bbox = bbox

class Tracker:
    def __init__(self):
        self.active_tracks : List[Track] = []
        self.stored_tracks : List[Track] = []
        self._max_inactive_time = 5.0  # seconds
        self._max_real_dist = 10.0  # meters
        self._max_stat_dist = 0.921  # chi2 3 dof 0.99

        self._measurements : PriorityQueue[Measurement] = PriorityQueue()
        self._tracker_thread : Thread = None
        self._lock = Lock()

        self.start_tracker()

    def start_tracker(self):
        if not self._running:
            self._running = True
            self._tracker_thread = Thread(target=self._tracker_loop)
            self._tracker_thread.start()
            print("Tracker started.")
        else:
            print("Tracker already running.")

    def stop_tracker(self):
        """ Stops the tracker and clears all tracks. """
        if self._running:
            self._running = False
            self._tracker_thread.join()
            print("Tracker stopped.")
        else:
            print("Tracker is not running.")

    def update(self, measurement : Measurement):
        """ Adds a new measurement to the queue for processing. """
        self._measurements.put((measurement.timestamp, measurement))