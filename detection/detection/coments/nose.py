import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped
import mediapipe as mp

class Nose(Node):
    def __init__(self):
        # Initialize the ROS2 node with the name 'nose_tf_node'
        super().__init__('nose_tf_node')

        # ------------------------- ROS SUBSCRIPTIONS -------------------------
        # Subscribe to the RGB image topic from the camera
        # Each callback is called when a new message is received
        self.rgb_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.rgb_callback, 10)

        # Subscribe to the depth image topic (for 3D position estimation)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_rect_raw', self.depth_callback, 10)

        # Subscribe to the camera intrinsics (focal lengths, principal point)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/color/camera_info', self.camera_info_callback, 10)

        # ------------------------- UTILITY INITIALIZATION -------------------------
        # Used to convert ROS Image messages <-> OpenCV image arrays
        self.bridge = CvBridge()

        # To store the latest received depth frame and camera parameters
        self.latest_depth = None
        self.camera_intrinsics = None

        # ------------------------- MEDIAPIPE POSE DETECTOR -------------------------
        # Initialize MediaPipe Pose solution for ONE human body keypoint detection
        self.mp_pose = mp.solutions.pose
        # Create a Pose object with a minimum detection confidence of 0.5
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5)
        # Confidence threshold to accept keypoints as valid
        self.conf_threshold = 0.5

        # ------------------------- TF2 BROADCASTER -------------------------
        # Used to publish dynamic transforms (e.g., human position relative to camera)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

    # =====================================================================
    # Callback for camera intrinsics — stores focal length and principal point
    # =====================================================================
    def camera_info_callback(self, msg):
        # Only save camera intrinsics once (they are static)
        if self.camera_intrinsics is not None:
            return

        # Extract intrinsic parameters from the camera matrix (K)
        # Matrix K = [fx  0  cx;
        #             0  fy  cy;
        #             0   0   1]
        self.camera_intrinsics = {
            'fx': msg.k[0],
            'fy': msg.k[4],
            'cx': msg.k[2],
            'cy': msg.k[5]
        }

    # =====================================================================
    # Callback for depth frames — store the most recent depth image
    # =====================================================================
    def depth_callback(self, msg):
        # Convert ROS Image (depth) message to a NumPy array (in mm)
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    # =====================================================================
    # Callback for RGB frames — detects human nose and publishes its TF
    # =====================================================================
    def rgb_callback(self, msg):
        # Ensure depth image and camera intrinsics are available before processing
        if self.latest_depth is None or self.camera_intrinsics is None:
            return

        # Convert ROS Image message to OpenCV format (BGR)
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Convert to RGB since MediaPipe expects RGB input
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run MediaPipe Pose detection
        results = self.pose.process(rgb_frame)

        # Copy frame for visualization
        display_frame = frame.copy()

        # Check if pose landmarks were successfully detected
        if results.pose_landmarks:
            # Extract the nose landmark (index 0 in MediaPipe PoseLandmark enum)
            nose = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            # Visibility confidence (0.0 - 1.0)
            confidence = nose.visibility

            # Process only if nose detection is confident enough
            if confidence >= self.conf_threshold:
                h, w, _ = frame.shape
                # Convert normalized landmark coordinates (0–1) to pixel coordinates
                cx, cy = int(nose.x * w), int(nose.y * h)

                # Extract a small window around the detected nose pixel in depth image
                # to reduce noise using median filtering
                depth_window = self.latest_depth[max(0, cy-5):cy+5, max(0, cx-5):cx+5]
                if depth_window.size == 0:
                    return

                # Compute the median depth value in this region (in millimeters)
                depth = float(np.median(depth_window))

                # Ignore invalid or missing depth readings
                if depth == 0.0 or np.isnan(depth):
                    return

                # Convert pixel coordinates (cx, cy, depth) → 3D camera coordinates
                X, Y, Z = self.pixel_to_3d(cx, cy, depth)

                # -------------------- PUBLISH TRANSFORM (TF) --------------------
                # Create a TransformStamped message for broadcasting human pose
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()  # current time
                t.header.frame_id = "camera_link"  # parent frame (camera)
                t.child_frame_id = "human_0"       # child frame (detected human)

                # Translation (convert from mm to meters)
                t.transform.translation.x = X / 1000.0
                t.transform.translation.y = Y / 1000.0
                t.transform.translation.z = Z / 1000.0

                # No orientation info from a single landmark → set to identity quaternion
                t.transform.rotation.w = 1.0

                # Broadcast transform (used by TF2 for tracking)
                self.tf_broadcaster.sendTransform(t)

                # -------------------- VISUALIZATION --------------------
                # Draw a green circle at the nose position
                cv2.circle(display_frame, (cx, cy), 5, (0, 255, 0), -1)
                # Display detection confidence
                cv2.putText(display_frame, f'Conf: {confidence:.2f}', (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the annotated frame in a window for debugging
        cv2.imshow("MediaPipe Human Detection", display_frame)
        cv2.waitKey(1)

    # =====================================================================
    # Convert 2D pixel + depth → 3D camera coordinates
    # Using the pinhole camera model:
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy
    # Z = depth
    # =====================================================================
    def pixel_to_3d(self, cx, cy, depth):
        fx = self.camera_intrinsics['fx']
        fy = self.camera_intrinsics['fy']
        cx_intr = self.camera_intrinsics['cx']
        cy_intr = self.camera_intrinsics['cy']

        X = (cx - cx_intr) * depth / fx
        Y = (cy - cy_intr) * depth / fy
        Z = depth
        return X, Y, Z

# =====================================================================
# MAIN ENTRY POINT
# =====================================================================
def main(args=None):
    # Initialize ROS2 communications
    rclpy.init(args=args)

    # Instantiate the node
    node = Nose()

    try:
        # Keep the node alive and processing callbacks
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Graceful shutdown on Ctrl+C
        pass
    finally:
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        # Shutdown ROS2
        rclpy.shutdown()

if __name__ == '__main__':
    main()
