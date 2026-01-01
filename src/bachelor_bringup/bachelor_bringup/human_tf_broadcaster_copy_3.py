import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped
import mediapipe as mp

class HumanTFNode(Node):
    def __init__(self):
        super().__init__('human_tf_node')

        # ROS subscriptions
        self.rgb_sub = self.create_subscription(Image, '/camera/color/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera/color/camera_info', self.camera_info_callback, 10)

        # CV Bridge
        self.bridge = CvBridge()
        self.latest_depth = None
        self.camera_intrinsics = None

        # MediaPipe Pose detector
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5)  # default, adjustable
        self.conf_threshold = 0.5  # default, can be modified via parameter

        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

    def camera_info_callback(self, msg):
        if self.camera_intrinsics is not None:
            return
        self.camera_intrinsics = {
            'fx': msg.k[0],
            'fy': msg.k[4],
            'cx': msg.k[2],
            'cy': msg.k[5]
        }

    def depth_callback(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def rgb_callback(self, msg):
        if self.latest_depth is None or self.camera_intrinsics is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        display_frame = frame.copy()

        if results.pose_landmarks:
            # Use the nose keypoint as representative human position
            nose = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            confidence = nose.visibility  # between 0 and 1

            if confidence >= self.conf_threshold:
                h, w, _ = frame.shape
                cx, cy = int(nose.x * w), int(nose.y * h)

                # Median depth around nose
                depth_window = self.latest_depth[max(0, cy-5):cy+5, max(0, cx-5):cx+5]
                if depth_window.size == 0:
                    return
                depth = float(np.median(depth_window))
                if depth == 0.0 or np.isnan(depth):
                    return

                X, Y, Z = self.pixel_to_3d(cx, cy, depth)

                # Publish TF
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = "camera_link"
                t.child_frame_id = "human_0"
                t.transform.translation.x = X/1000.0  # Convert mm to meters
                t.transform.translation.y = Y/1000.0
                t.transform.translation.z = Z/1000.0
                t.transform.rotation.w = 1.0
                self.tf_broadcaster.sendTransform(t)

                # Draw detection on frame
                cv2.circle(display_frame, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(display_frame, f'Conf: {confidence:.2f}', (cx+10, cy-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("MediaPipe Human Detection", display_frame)
        cv2.waitKey(1)

    def pixel_to_3d(self, cx, cy, depth):
        fx = self.camera_intrinsics['fx']
        fy = self.camera_intrinsics['fy']
        cx_intr = self.camera_intrinsics['cx']
        cy_intr = self.camera_intrinsics['cy']

        X = (cx - cx_intr) * depth / fx
        Y = (cy - cy_intr) * depth / fy
        Z = depth
        return X, Y, Z

def main(args=None):
    rclpy.init(args=args)
    node = HumanTFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
