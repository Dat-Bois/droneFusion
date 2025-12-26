import numpy as np
from typing import List, Tuple
from filterpy.kalman import KalmanFilter, predict

'''
Every object has its own track / filter.
consists of:
 - state [x, y, z, vx, vy, vz, ax, ay, az] (units are in meters and seconds)
 - measurement [3x1] (x y z)
 - process noise is on accel term only but propagated to others
 - covariance
 - maturity

and methods to:
 - update state with new measurement
 - get current state estimate
'''

class PoseStamped:
    def __init__(self, timestamp : float, state : np.ndarray, mahalonobis : float, dist : float):
        self.timestamp = timestamp
        self.state = state.copy()
        self.mahalonobis = mahalonobis
        self.dist = dist

    def __str__(self):
        return f"PoseStamped: \n \
        t={self.timestamp}, \n \
        mahalonobis={self.mahalonobis}, \n \
        dist={self.dist} \n \
        state={self.state.flatten()}"

class Track:
    def __init__(self, initial_measurement : np.ndarray, timestamp : float):
        self.kf = KalmanFilter(dim_x=9, dim_z=3)
        # Measurement Matrix
        self.kf.H = np.array([  [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0, 0, 0]])
        # Initial State
        self.kf.x[:3] = initial_measurement.reshape(3,1)
        self.kf.x[3:] = 0.0
        # Covariance Matrix
        self.kf.P *= 2
        # Process Noise (only acceleration noise)
        self.Q = np.eye(3) * 0.91
        # Measurement Noise
        self.kf.R *= 5.0

        self.last_update = timestamp  # time since last update
        self.maturity = 0     # number of consecutive updates
        self._mature_threshold = 5

        self.history : List[PoseStamped] = []

    def stat_dists(self, Z : np.ndarray, R : np.ndarray, timestamp : float) -> Tuple[float, float]:
        """ Returns the Mahalanobis distance and "real dist" between the measurement and the predicted state. """
        F = self._state_transition(timestamp)
        Q = self._process_transition(timestamp)
        X_pred, P_pred = predict(self.kf.x, self.kf.P, F, Q)
        X_pred = self.kf.H @ X_pred
        xpv = (P_pred[0,0] + P_pred[0,3] + P_pred[3,3])
        ypv = (P_pred[1,1] + P_pred[1,4] + P_pred[4,4])
        zpv = (P_pred[2,2] + P_pred[2,5] + P_pred[5,5])
        res = (Z - X_pred)**2
        sd = (res[0,0] / (xpv + R[0,0])) + (res[1,0] / (ypv + R[1,1])) + (res[2,0] / (zpv + R[2,2]))
        rd = ((res[0,0]*xpv)/R[0,0]) + ((res[1,0]*ypv)/R[1,1]) + ((res[2,0]*zpv)/R[2,2])
        return sd, rd
        
    def is_mature(self) -> bool:
        return self.maturity >= self._mature_threshold

    def _state_transition(self, timestamp : float):
        dt = timestamp - self.last_update
        F =    np.array([[1, 0, 0, dt, 0, 0, 0.5*dt**2, 0, 0],
                [0, 1, 0, 0, dt, 0, 0, 0.5*dt**2, 0],
                [0, 0, 1, 0, 0, dt, 0, 0, 0.5*dt**2],
                [0, 0, 0, 1, 0, 0, dt, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, dt, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, dt],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        return F
    
    def _process_transition(self, timestamp : float):
        # Propogates the process noise matrix based on time elapsed
        # and returns the new process noise matrix Q
        dt = timestamp - self.last_update
        G =    np.array([[0.5*dt**2, 0, 0],
                [0, 0.5*dt**2, 0],
                [0, 0, 0.5*dt**2],
                [dt, 0, 0],
                [0, dt, 0],
                [0, 0, dt],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])
        # 9x3 @ 3x3 @ 3x9 = 9x9
        return G @ self.Q @ G.T

    def predict(self, timestamp : float):
        F = self._state_transition(timestamp)
        Q = self._process_transition(timestamp)
        return predict(self.kf.x, self.kf.P, F, Q)

    def update(self, Z : np.ndarray, R : np.ndarray, timestamp : float):
        Z = Z.reshape(3,1)
        self.kf.R = R
        if self.is_mature():
            m, r = self.stat_dists(Z, R, timestamp)
            self.history.append(PoseStamped(timestamp, self.kf.x, m, r))
        self.kf.F = self._state_transition(timestamp)
        self.kf.Q = self._process_transition(timestamp)
        self.kf.predict()
        if self.maturity == 0:
            # give some initial velocity estimate
            dt = timestamp - self.last_update
            if dt > 0:
                vel_est = (Z - self.kf.x[:3]) / dt
                self.kf.x[3:6] = vel_est
        self.kf.update(Z)
        self.maturity += 1
        self.last_update = timestamp