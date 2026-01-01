import math
import time
from typing import Dict, List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseArray, Pose, Point, PointStamped
from std_msgs.msg import Header

from social_force_layer.social_force_model import SocialForceModel


# --------------------------------------------------------------------
# Track object: holds state, covariance, timestamp
# --------------------------------------------------------------------
class Track:
    def __init__(self, x: float, y: float, time: float):
        # State vector: [px, py, vx, vy]
        self.x = np.array([x, y], dtype=float)

        self.stored = []

        # Last update time
        self.last_update = time

    def stored(self, vx: float, vy: float):
        return

    def predict_linear(self, dt: float, Q: np.ndarray):# this section might be wrong
        """Simple constant-velocity prediction."""
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ])

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q


# --------------------------------------------------------------------
# SocialForceEKF ROS Node
# --------------------------------------------------------------------
class SocialForceEKF(Node):
    def __init__(self):
        super().__init__('social_force_predictor_ekf')

        # ---------------- Parameters ----------------
        # Load parameters
        self.gating = 1.0               # max association distance
        self.Q_scale = 0.5              # process noise scale
        self.R_meas = 0.2               # measurement noise stddev
        self.track_timeout = 5.0        # seconds to delete stale tracks
        self.publish_rate = 15.0        # Hz to publish predictions
        self.prediction_horizon = 10    # steps to predict ahead 

        # Social Force Model
        self.sfm = SocialForceModel()

        # Track database
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1

        # ---------------- ROS Interfaces ----------------
        # change data type 
        self.create_subscription(PointStamped, "depth", self.depth_callback, 10)

        self.publisher = self.create_publisher(PoseArray, "predicted_humans", 10)

        # Prediction publishing timer########################
        self.create_timer(1.0 / self.publish_rate,
                          self._publish_predictions)

        self.get_logger().info("SocialForceEKF started.")


    # ----------------------------------------------------------------
    # Callbacks for incoming detections
    # ----------------------------------------------------------------
    def depth_callback(self, msg):
        # Convert (lateral=x, forward=z) into (x_forward, y_lateral)
        detections = [float(msg.point.z), float(msg.point.x), 
                      float(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)]
        self._process_detections(detections)


    # ----------------------------------------------------------------
    # Processing detections: prediction, association, update
    # ----------------------------------------------------------------
    def _process_detections(self, detections):
        now = detections[2]

        # --- 1. Predict all tracks ---
        for track in self.tracks.values():
            dt = max(0.01, now - track.last_update)
            Q = np.eye(4) * (self.Q_scale * dt)
            track.predict_linear(dt, Q)

        # --- 2. Data association (nearest neighbor) ---
        unmatched = set(range(len(detections))) #definatly wrong

        for i, det in enumerate(detections):
            best_tid, best_dist = None, float('inf')

            # Find nearest track
            for tid, track in self.tracks.items():
                px, py = track.x[0:2]
                dist = math.hypot(px - det[0], py - det[1])

                if dist < best_dist:
                    best_dist = dist
                    best_tid = tid

            # If detection within gating threshold → update track
            if best_tid is not None and best_dist <= self.gating:
                self._update_track(self.tracks[best_tid], det)
                unmatched.discard(i)

        # --- 3. Create new tracks for unmatched detections ---
        for i in unmatched:
            x, y = detections[i]
            self.tracks[self._next_id] = Track(x, y, 0.0, 0.0)
            self._next_id += 1

        # --- 4. Delete stale tracks ---
        stale = [
            tid for tid, track in self.tracks.items()
            if now - track.last_update > self.track_timeout
        ]
        for tid in stale:
            del self.tracks[tid]


    # ----------------------------------------------------------------
    # Kalman Update Step
    # ----------------------------------------------------------------
    def _update_track(self, track: Track, det: Tuple[float, float]):
        z = np.array([det[0], det[1]])
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        R = np.eye(2) * (self.R_meas ** 2)

        # Innovation
        z_pred = H @ track.x
        y = z - z_pred

        # Kalman gain
        S = H @ track.P @ H.T + R
        K = track.P @ H.T @ np.linalg.inv(S)

        # State update
        track.x = track.x + K @ y

        # Covariance update
        I = np.eye(4)
        track.P = (I - K @ H) @ track.P

        track.last_update = time.time()


    # ----------------------------------------------------------------
    # Publishing predicted future positions (SFM rollout)
    # ----------------------------------------------------------------
    def _publish_predictions(self):
        msg = PoseArray()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_link'

        others = [tuple(track.x[0:2]) for track in self.tracks.values()]

        for track in self.tracks.values():
            pos = track.x[:2].copy()
            vel = track.x[2:].copy()
            goal = pos + vel  # simple projected goal

            sim_pos = pos
            sim_vel = vel

            # Run SFM simulation for fixed number of future steps
            for _ in range(self.prediction_horizon):
                try:
                    sim_pos, sim_vel = self.sfm.step(sim_pos, sim_vel, goal, others)
                except Exception:
                    dt = getattr(self.sfm, "dt", 0.1)
                    sim_pos = sim_pos + sim_vel * dt

                p = Pose()
                p.position.x = float(sim_pos[0])
                p.position.y = float(sim_pos[1])
                msg.poses.append(p)

        self.publisher.publish(msg)


# --------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = SocialForceEKF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
