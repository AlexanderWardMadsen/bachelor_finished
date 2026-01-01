"""EKF-based social force predictor node

This node subscribes to detections (PoseArray on /tracker_humans), runs a simple
multi-target EKF where the motion prediction uses the SocialForceModel
from the `social_force_layer` package, and publishes predicted positions
as a PoseArray on /predicted_humans for the costmap layer.

Notes:
- This is a lightweight example intended to integrate the SFM as a
  motion predictor inside the EKF predict step. The EKF covariance
  prediction is approximated by adding process noise (no linearized F).
- Data association uses nearest-neighbor with a configurable gating
  distance. New tracks are created for unmatched detections.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose, Point
from std_msgs.msg import Header

from social_force_layer.social_force_model import SocialForceModel


class Track:
    def __init__(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0):
        # state vector [px, py, vx, vy]
        self.x = np.array([x, y, vx, vy], dtype=float)
        self.P = np.diag([0.5, 0.5, 1.0, 1.0]) # initial covariance
        self.last_update = time.time()

    def predict_linear(self, dt: float, Q: np.ndarray):
        # simple constant velocity linear predict for covariance update
        F = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q


class SocialForceEKF(Node):
    def __init__(self):
        super().__init__('social_force_predictor_ekf')
        # parameters
        self.declare_parameter('detection_topic', '/humans_raw')
        # optional: detection publishes Point messages (x,y-id,z) on this topic
        # the detection package's HumanTFPublisher publishes a Point on 'depth'
        # with: point.x = lateral (m), point.y = id, point.z = forward (m)
        # We convert that -> Pose where position.x = forward (point.z), position.y = lateral (point.x)
        self.declare_parameter('detection_point_topic', 'depth')
        self.declare_parameter('predicted_topic', '/predicted_humans')
        self.declare_parameter('gating_distance', 1.0)
        self.declare_parameter('process_noise', 0.5)
        self.declare_parameter('measurement_noise', 0.2)
        self.declare_parameter('track_timeout', 5.0)
        self.declare_parameter('publish_rate', 2.0)
        self.declare_parameter('prediction_horizon', 10) # steps to predict ahead

        self.detection_topic = self.get_parameter('detection_topic').value
        self.detection_point_topic = self.get_parameter('detection_point_topic').value
        self.predicted_topic = self.get_parameter('predicted_topic').value
        self.gating = float(self.get_parameter('gating_distance').value)
        self.Q_scale = float(self.get_parameter('process_noise').value)
        self.R_meas = float(self.get_parameter('measurement_noise').value)
        self.track_timeout = float(self.get_parameter('track_timeout').value)
        self.publish_rate = float(self.get_parameter('publish_rate').value)
        self.prediction_horizon = int(self.get_parameter('prediction_horizon').value)

        # SFM model used in predict step
        self.sfm = SocialForceModel()

        # tracks, id -> Track
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1

        # ROS interfaces
        # Subscribe to PoseArray detections (legacy/tracker) if used
        self.sub = self.create_subscription(PoseArray, self.detection_topic, self._detections_cb, 10)
        # Optionally subscribe directly to detection Point messages (e.g. detection publishes Point on 'depth')
        if self.detection_point_topic:
            try:
                self.sub_point = self.create_subscription(Point, self.detection_point_topic, self._point_cb, 10)
                self.get_logger().info(f"Also subscribing to detection Point topic: {self.detection_point_topic}")
            except Exception:
                # best-effort: log and continue
                self.get_logger().warning(f"Failed to create Point subscription for {self.detection_point_topic}")
        self.pub = self.create_publisher(PoseArray, self.predicted_topic, 10)

        self.create_timer(1.0 / max(self.publish_rate, 1e-6), self._publish_predictions)

        self.get_logger().info('SocialForceEKF started')

    # ---------------- callbacks ----------------
    def _detections_cb(self, msg: PoseArray) -> None:
        # Convert PoseArray -> list of (x,y) and hand off to processor
        detections = [(float(p.position.x), float(p.position.y)) for p in msg.poses]
        if not detections:
            return
        self._process_detections(detections)

    def _point_cb(self, msg: Point) -> None:
        # Detection node publishes a Point where:
        #  - msg.x = lateral (m)
        #  - msg.y = person id (or small int)
        #  - msg.z = forward depth (m)
        # Convert to EKF (x_forward, y_lateral) ordering used elsewhere
        try:
            x_forward = float(msg.z)
            y_lateral = float(msg.x)
        except Exception:
            return
        detections = [(x_forward, y_lateral)]
        self._process_detections(detections)

    def _process_detections(self, detections: List[Tuple[float, float]]) -> None:
        now = time.time()

        # Predict all tracks to current time using simple linear predict
        for tid, track in list(self.tracks.items()):
            dt = now - track.last_update
            if dt <= 0:
                dt = 0.01
            Q = np.eye(4) * (self.Q_scale * dt)
            track.predict_linear(dt, Q)

        # Data association, nearest neighbor
        unmatched_dets = set(range(len(detections)))
        assigned_tracks = set()

        for det_idx, det in enumerate(detections):
            dx = []
            for tid, track in self.tracks.items():
                pred_pos = track.x[0:2]
                dist = math.hypot(pred_pos[0] - det[0], pred_pos[1] - det[1])
                dx.append((dist, tid))
            if dx:
                dx.sort()
                dist, best_tid = dx[0]
                if dist <= self.gating:
                    # update best_tid with detection
                    self._update_track(self.tracks[best_tid], det)
                    assigned_tracks.add(best_tid)
                    unmatched_dets.discard(det_idx)

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            x, y = detections[det_idx]
            t = Track(x, y, 0.0, 0.0)
            self.tracks[self._next_id] = t
            self._next_id += 1

        # remove stale tracks
        to_delete = []
        for tid, track in self.tracks.items():
            if now - track.last_update > self.track_timeout:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

    def _update_track(self, track: Track, det: Tuple[float, float]) -> None:
        z = np.array([det[0], det[1]])
        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        R = np.eye(2) * (self.R_meas ** 2)

        # Measurement prediction
        z_pred = H @ track.x
        S = H @ track.P @ H.T + R
        K = track.P @ H.T @ np.linalg.inv(S)
        y = z - z_pred
        track.x = track.x + K @ y
        track.P = (np.eye(4) - K @ H) @ track.P
        track.last_update = time.time()

    # ---------------- publish predicted positions ----------------
    def _publish_predictions(self) -> None:
        # Before publishing, do a short SFM-based prediction step for each track
        predicted_msg = PoseArray()
        predicted_msg.header = Header()
        predicted_msg.header.stamp = self.get_clock().now().to_msg()
        predicted_msg.header.frame_id = 'map'

        # collect other positions for SFM
        others = [tuple(track.x[0:2]) for track in self.tracks.values()]

        for tid, track in self.tracks.items():
            pos = np.array([track.x[0], track.x[1]])
            vel = np.array([track.x[2], track.x[3]])
            # approximate a goal, a short projection along current velocity
            goal = pos + vel * 1.0
            # simulate multiple SFM steps for a short horizon and publish all points
            sim_pos = pos.copy()
            sim_vel = vel.copy()
            for step in range(self.prediction_horizon):
                try:
                    sim_pos, sim_vel = self.sfm.step(sim_pos, sim_vel, goal, others)
                except Exception:
                    # fallback to linear small step
                    sim_pos = sim_pos + sim_vel * (self.sfm.dt if hasattr(self.sfm, 'dt') else 0.1)
                p = Pose()
                p.position.x = float(sim_pos[0])
                p.position.y = float(sim_pos[1])
                p.position.z = 0.0
                predicted_msg.poses.append(p)

        self.pub.publish(predicted_msg)


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
