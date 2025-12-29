import time
import numpy as np
from typing import List, Tuple
from queue import PriorityQueue, Empty
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
        - If measurement works with multiple tracks, choose the one with lowest stat dist (this can be expanded to MHT later)
        - Update track with measurement

Callable functions:
    - update
    - get_latest_tracks (returns the latest state and bbox for each mature active track) (PoseStamped)
    - get_active_tracks
    - get_stored_tracks
'''

class Measurement:
    def __init__(self, timestamp : float, pose : np.ndarray, R : np.ndarray = np.eye(3), bbox: List[float] = [1,1]):
        self.timestamp = timestamp # float seconds
        self.pose = pose # np.ndarray shape (3,) in meters [x, y, z]
        self.R = R # np.ndarray shape (3,3) measurement noise
        self.bbox = bbox # List[float] shape (2,) in meters [width, height]

    def copy(self):
        return Measurement(self.timestamp, self.pose.copy(), self.R.copy(), self.bbox.copy())

class Tracker:
    def __init__(self):
        self.active_tracks : List[Track] = []
        self.stored_tracks : List[Track] = []
        self._max_inactive_time = 5.0  # seconds
        self._max_real_dist = 10.0  # meters
        self._max_stat_dist = 0.921  # chi2 3 dof 0.99

        self._last_measurement_time = None
        self._measurements : PriorityQueue[Tuple[float, Measurement]] = PriorityQueue()
        self._tracker_thread : Thread = None
        self._lock = Lock()

        self._running = False
        self.start_tracker()

    def start_tracker(self):
        with self._lock:
            if not self._running:
                self._running = True
                self._tracker_thread = Thread(target=self._tracker_loop)
                self._tracker_thread.start()
                print("Tracker started.")
            else:
                print("Tracker already running.")

    def stop_tracker(self):
        """ Stops the tracker and clears all tracks. """
        self._running = False
        if self._tracker_thread is not None:
            self._tracker_thread.join()
            self._tracker_thread = None
            print("Tracker stopped.")
        else:
            print("Tracker is not running.")

    def update(self, measurement : Measurement):
        """ Adds a new measurement to the queue for processing. """
        self._measurements.put((measurement.timestamp, measurement))

    def get_latest_tracks(self, timestamp : float = None) -> List[PoseStamped]:
        """ Returns the latest PoseStamped for each mature active track. """
        with self._lock:
            latest_tracks = []
            for track in self.active_tracks:
                if track.is_mature():
                    pose_stamped = track.get_state(timestamp)
                    latest_tracks.append(pose_stamped)
            return latest_tracks
        
    def get_active_tracks(self) -> List[Track]:
        """ Returns the list of active tracks. """
        with self._lock:
            return list(self.active_tracks)
        
    def get_stored_tracks(self) -> List[Track]:
        """ Returns the list of stored tracks. """
        with self._lock:
            return list(self.stored_tracks)
        
    def _ageout_tracks(self, current_time : float):
        """ Moves inactive mature tracks to stored tracks. """
        if current_time is None: return
        to_remove : set[Track] = set()
        for track in self.active_tracks:
            if (current_time - track.last_update) > self._max_inactive_time:
                if track.is_mature():
                    self.stored_tracks.append(track)
                to_remove.add(track)
        if len(to_remove) > 0:
            self.active_tracks = [t for t in self.active_tracks if t not in to_remove]
        
    def _tracker_loop(self):
        """ Main tracker loop running in a separate thread. """
        while self._running:
            try:
                timestamp, measurement = self._measurements.get(timeout=1.0)
            except Empty:
                with self._lock:
                    self._ageout_tracks(self._last_measurement_time)
                time.sleep(0.1)
                continue
            self._last_measurement_time = timestamp if self._last_measurement_time is None else max(self._last_measurement_time, timestamp)
            with self._lock:
                self._ageout_tracks(self._last_measurement_time)
                best_track = None
                for track in self.active_tracks:
                    m_dist, r_dist = track.stat_dists(measurement.pose, measurement.R, timestamp)
                    if m_dist < self._max_stat_dist and r_dist < self._max_real_dist:
                        if best_track is None:
                            best_track = (m_dist, r_dist, track)
                        else:
                            if m_dist < best_track[0]:
                                best_track = (m_dist, r_dist, track)
                if best_track is not None:
                    best_track = best_track[2]
                    best_track.update(measurement.pose, measurement.R, timestamp, measurement.bbox)
                else:
                    new_track = Track(measurement.pose, timestamp)
                    new_track.update(measurement.pose, measurement.R, timestamp, measurement.bbox)
                    self.active_tracks.append(new_track)