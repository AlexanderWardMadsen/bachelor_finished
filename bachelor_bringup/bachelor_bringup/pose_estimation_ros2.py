#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import mediapipe as mp
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def angle_between_points(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return None
    cos_angle = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def compute_joint_angles(landmarks, width, height):
    def get_point(idx):
        lm = landmarks[idx]
        return int(lm.x * width), int(lm.y * height)

    angles = {}
    try:
        left_shoulder = 11
        left_elbow = 13
        left_wrist = 15
        right_shoulder = 12
        right_elbow = 14
        right_wrist = 16
        left_hip = 23
        left_knee = 25
        left_ankle = 27
        right_hip = 24
        right_knee = 26
        right_ankle = 28

        angles["left_elbow"] = angle_between_points(
            get_point(left_shoulder), get_point(left_elbow), get_point(left_wrist)
        )
        angles["right_elbow"] = angle_between_points(
            get_point(right_shoulder), get_point(right_elbow), get_point(right_wrist)
        )
        angles["left_knee"] = angle_between_points(
            get_point(left_hip), get_point(left_knee), get_point(left_ankle)
        )
        angles["right_knee"] = angle_between_points(
            get_point(right_hip), get_point(right_knee), get_point(right_ankle)
        )
    except Exception:
        pass

    return {k: round(v, 1) if v else None for k, v in angles.items()}


class PoseEstimationNode(Node):
    def __init__(self):
        super().__init__('pose_estimation_node')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10
        )
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.get_logger().info("Pose Estimation Node started, subscribing to /camera/color/image_raw")

    def image_callback(self, msg: Image):
        # Convert ROS Image -> OpenCV
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV bridge error: {e}")
            return

        # Flip for mirror effect
        frame = cv2.flip(cv_image, 1)
        h, w = frame.shape[:2]

        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe Pose
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

            angles = compute_joint_angles(results.pose_landmarks.landmark, w, h)
            y0 = 30
            for i, (name, angle) in enumerate(angles.items()):
                if angle is not None:
                    cv2.putText(frame, f"{name}: {angle:.1f}", (10, y0 + i * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show the frame
        cv2.imshow("ROS2 Pose Estimation (press 'q' to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
