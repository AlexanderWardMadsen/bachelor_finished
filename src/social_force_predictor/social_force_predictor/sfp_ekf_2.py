#!/usr/bin/env python3
"""
EKF Multi-Target Tracker Node

Tracks multiple people using Extended Kalman Filter with constant velocity model.
Subscribes to raw detections and publishes filtered state with velocity estimates.
"""

import math
import time
from typing import Dict, List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose, Point, PointStamped
from std_msgs.msg import Header



class Track:
    def __init__(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0):
        self.x = np.array([x, y, vx, vy], dtype=float) # state vector [px, py, vx, vy]
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
        self.x = F @ self.x # Predicts the next state.
        self.P = F @ self.P @ F.T + Q # Propagates the covariance, adding process noise Q.



class SocialForceEKF(Node):
    def __init__(self):
        super().__init__('sf_ekf')
        # parameters
        # Detection publishes Point messages on 'depth' topic
        # with: point.x = lateral (m), point.y = id, point.z = forward (m)
        self.declare_parameter('detection_point_topic', 'depth')
        self.declare_parameter('ekf_topic', '/ekf')
        self.declare_parameter('gating_distance', 1.0)
        self.declare_parameter('process_noise', 0.5) # higher = smoother but slower
        self.declare_parameter('measurement_noise', 0.2) # how strongly to trust measurements
        self.declare_parameter('track_timeout', 5.0) # seconds to delete stale tracks
        self.declare_parameter('publish_rate', 1.0) # Hz

        self.detection_point_topic = self.get_parameter('detection_point_topic').value
        self.ekf_topic = self.get_parameter('ekf_topic').value
        self.gating = float(self.get_parameter('gating_distance').value)
        self.Q_scale = float(self.get_parameter('process_noise').value)
        self.R_meas = float(self.get_parameter('measurement_noise').value)
        self.track_timeout = float(self.get_parameter('track_timeout').value)
        self.publish_rate = float(self.get_parameter('publish_rate').value)

        # tracks, id -> Track
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1

        # ROS interfaces - subscribe directly to detection Point messages
        self.sub_point = self.create_subscription(PointStamped, self.detection_point_topic, self._point_cb, 10)
        self.pub = self.create_publisher(PoseArray, self.ekf_topic, 10)

        self.create_timer(1.0 / max(self.publish_rate, 1e-6), self._publish_filtered_state)

        self.get_logger().info(f'SocialForceEKF: subscribing to {self.detection_point_topic}, publishing to {self.ekf_topic}')



    # ---------------- callback ----------------
    def _point_cb(self, msg: PointStamped) -> None:
        # Detection node publishes a Point where:
        #  - msg.x = lateral (m)
        #  - msg.y = person id (or small int)
        #  - msg.z = forward depth (m)
        try:
            x_forward = float(msg.point.z)
            y_lateral = float(msg.point.x)
            self._process_detections([(x_forward, y_lateral)])
        except Exception:
            return



    def _process_detections(self, detections: List[Tuple[float, float]]) -> None: #the fucks a tuple
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

        for det_idx, det in enumerate(detections):
            dx = []
            for tid, track in self.tracks.items():
                self.get_logger().info(f'{track.x} and {tid}')
                pred_pos = track.x[0:2]
                dist = math.hypot(pred_pos[0] - det[0], pred_pos[1] - det[1])
                dx.append((dist, tid))
            if dx:
                dx.sort()
                dist, best_tid = dx[0]
                if dist <= self.gating:
                    # update best_tid with detection
                    self._update_track(self.tracks[best_tid], det)
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
        z_pred = H @ track.x # predicted measurement
        S = H @ track.P @ H.T + R # innovation covariance
        K = track.P @ H.T @ np.linalg.inv(S) # Kalman gain
        y = z - z_pred # measurement residual
        track.x = track.x + K @ y # state update
        track.P = (np.eye(4) - K @ H) @ track.P # covariance update
        track.last_update = time.time() # timestamp update



    # ---------------- publish current filtered state ----------------
    def _publish_filtered_state(self) -> None:
        # Publish current filtered positions and velocities (no prediction)
        filtered_msg = PoseArray()
        filtered_msg.header = Header()
        filtered_msg.header.stamp = self.get_clock().now().to_msg()
        filtered_msg.header.frame_id = 'camera_link'

        for tid, track in self.tracks.items():
            p = Pose()
            # Current position
            p.position.x = float(track.x[0])
            p.position.y = float(track.x[1])
            p.position.z = 0.0
            # Velocity encoded in orientation for predictor to use
            p.orientation.x = float(track.x[2]) # vx
            p.orientation.y = float(track.x[3]) # vy
            p.orientation.z = 0.0
            p.orientation.w = 1.0
            filtered_msg.poses.append(p)

        self.pub.publish(filtered_msg)



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
