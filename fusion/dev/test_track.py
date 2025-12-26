import numpy as np
import unittest
import matplotlib.pyplot as plt
from fusion.track import Track, PoseStamped

class TestTrackLogic(unittest.TestCase):
    def setUp(self):
        """Setup a track instance before every test"""
        self.start_pos = np.array([10.0, 20.0, 5.0])
        self.t0 = 0.0
        self.tracker = Track(self.start_pos, self.t0)
        self.R_mock = np.eye(3) * 0.1 # Standard measurement noise

    def test_initialization(self):
        """Verify initial state is set correctly"""
        np.testing.assert_array_almost_equal(
            self.tracker.kf.x[:3].flatten(), 
            self.start_pos, 
            err_msg="Initial position in filter does not match input"
        )
        # Check maturity is 0
        self.assertEqual(self.tracker.maturity, 0)
        self.assertFalse(self.tracker.is_mature())

    def test_maturity_growth(self):
        """Verify maturity increases and triggers flag"""
        dt = 0.1
        for i in range(1, 6):
            meas = self.start_pos + i  # Fake movement
            self.tracker.update(meas, self.R_mock, self.t0 + i*dt)
            self.assertEqual(self.tracker.maturity, i)
        self.assertTrue(self.tracker.is_mature())

    def test_history_population(self):
        """Verify history is only populated AFTER maturity"""
        dt = 0.1
        for i in range(1, 5):
            self.tracker.update(self.start_pos, self.R_mock, self.t0 + i*dt)
        self.assertEqual(len(self.tracker.history), 0, "History populated before maturity!")
        self.tracker.update(self.start_pos, self.R_mock, self.t0 + 5*dt)
        self.tracker.update(self.start_pos, self.R_mock, self.t0 + 6*dt)
        self.assertEqual(len(self.tracker.history), 1, "History failed to populate after maturity")
        latest = self.tracker.history[-1]
        self.assertIsInstance(latest, PoseStamped)
        self.assertEqual(latest.state.shape, (9, 1))

def run_simulation():
    print("\n--- Running 3D Particle Simulation ---")
    t_steps = np.linspace(0, 10, 100) # seconds, steps
    radius = 20
    gt_x = radius * np.cos(t_steps)
    gt_y = radius * np.sin(t_steps)
    gt_z = t_steps * 2 # Rise 2m/s
    
    noise_std = 0.5
    meas_x = gt_x + np.random.normal(0, noise_std, size=len(t_steps))
    meas_y = gt_y + np.random.normal(0, noise_std, size=len(t_steps))
    meas_z = gt_z + np.random.normal(0, noise_std, size=len(t_steps))
    
    initial_meas = np.array([meas_x[0], meas_y[0], meas_z[0]])
    tracker = Track(initial_meas, t_steps[0])
    
    R = np.eye(3) * (noise_std**2)
    
    track_history = []
    
    for i in range(1, len(t_steps)):
        curr_time = t_steps[i]
        Z = np.array([meas_x[i], meas_y[i], meas_z[i]])
        tracker.update(Z, R, curr_time)
        pos = tracker.kf.x[:3].flatten()
        track_history.append(pos)

    track_history = np.array(track_history)
    
    gt_aligned = np.stack([gt_x[1:], gt_y[1:], gt_z[1:]], axis=1)
    meas_aligned = np.stack([meas_x[1:], meas_y[1:], meas_z[1:]], axis=1)
    
    meas_error = np.linalg.norm(meas_aligned - gt_aligned, axis=1).mean()
    track_error = np.linalg.norm(track_history - gt_aligned, axis=1).mean()
    
    print(f"Average Measurement Error (Noise): {meas_error:.4f} m")
    print(f"Average Tracker Error (Filtered):    {track_error:.4f} m")
    print(f"Improvement: {(meas_error - track_error)/meas_error * 100:.1f}%")

    fig = plt.figure(figsize=(12, 6))
    
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot(gt_x, gt_y, gt_z, 'k--', label='Ground Truth', linewidth=2, alpha=0.6)
    ax.scatter(meas_x, meas_y, meas_z, c='r', s=10, alpha=0.3, label='Noisy Measurements')
    ax.plot(track_history[:,0], track_history[:,1], track_history[:,2], 'b-', linewidth=2, label='Filter Track')
    
    ax.set_title("3D Target Tracking (Helix)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(gt_x, gt_y, 'k--', label='Ground Truth')
    ax2.scatter(meas_x, meas_y, c='r', s=10, alpha=0.3, label='Measurements')
    ax2.plot(track_history[:,0], track_history[:,1], 'b-', linewidth=2, label='Filter Track')
    ax2.set_title("Top-Down View (X-Y)")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

    # print("\n--- Tracker History Samples ---")
    # for pose in tracker.history:
    #     print(pose)

def run_simulation_linear():
    print("\n--- Running Linear/Parabolic Simulation (Projectile) ---")
    
    dt = 0.2
    t_steps = np.arange(0, 10, dt)

    vx_true = 15.0
    vy_true = 5.0
    vz_true = 40.0 # Initial upward velocity
    g = -9.81      # Gravity
    
    gt_x = vx_true * t_steps
    gt_y = vy_true * t_steps
    gt_z = (vz_true * t_steps) + (0.5 * g * t_steps**2)

    noise_std = 2.0 
    meas_x = gt_x + np.random.normal(0, noise_std, size=len(t_steps))
    meas_y = gt_y + np.random.normal(0, noise_std, size=len(t_steps))
    meas_z = gt_z + np.random.normal(0, noise_std, size=len(t_steps))
    
    initial_meas = np.array([meas_x[0], meas_y[0], meas_z[0]])
    tracker = Track(initial_meas, t_steps[0])
    
    track_history = []
    cov_history = [] 
    R = np.eye(3) * (noise_std**2)
    
    for i in range(1, len(t_steps)):
        curr_time = t_steps[i]
        Z = np.array([meas_x[i], meas_y[i], meas_z[i]])
        tracker.update(Z, R, curr_time)
        pos = tracker.kf.x[:3].flatten()
        track_history.append(pos)
        cov_history.append(np.diag(tracker.kf.P[:3,:3]))

    track_history = np.array(track_history)
    cov_history = np.array(cov_history)

    gt_aligned = np.stack([gt_x[1:], gt_y[1:], gt_z[1:]], axis=1)
    meas_aligned = np.stack([meas_x[1:], meas_y[1:], meas_z[1:]], axis=1)
    
    meas_error = np.linalg.norm(meas_aligned - gt_aligned, axis=1).mean()
    track_error = np.linalg.norm(track_history - gt_aligned, axis=1).mean()
    
    print(f"Average Measurement Error (Noise): {meas_error:.4f} m")
    print(f"Average Tracker Error (Filtered):    {track_error:.4f} m")
    print(f"Improvement: {(meas_error - track_error)/meas_error * 100:.1f}%")
    
    fig = plt.figure(figsize=(14, 6))
    
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot(gt_x[1:], gt_y[1:], gt_z[1:], 'k--', label='Ground Truth (Parabola)', linewidth=2)
    ax.scatter(meas_x[1:], meas_y[1:], meas_z[1:], c='r', s=10, alpha=0.2, label='Noisy Measurements')
    ax.plot(track_history[:,0], track_history[:,1], track_history[:,2], 'b-', linewidth=2, label='Filter Output')
    
    ax.set_title("3D Projectile Motion")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(t_steps[1:], np.sqrt(cov_history[:,0]), label='X Uncertainty (std dev)')
    ax2.plot(t_steps[1:], np.sqrt(cov_history[:,1]), label='Y Uncertainty (std dev)')
    ax2.plot(t_steps[1:], np.sqrt(cov_history[:,2]), label='Z Uncertainty (std dev)')
    ax2.set_title("Filter Convergence (P Matrix)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Standard Deviation (m)")
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    print("--- Running Unit Tests ---")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    run_simulation()
    run_simulation_linear()

    