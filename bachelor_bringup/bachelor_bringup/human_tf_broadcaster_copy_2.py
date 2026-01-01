import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped

class HumanTFNode(Node):
    def __init__(self):
        super().__init__('human_tf_node')

        # ROS subscriptions
        self.rgb_sub = self.create_subscription(Image, '/camera/color/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera/color/camera_info', self.camera_info_callback, 10)

        # CV Bridge
        self.bridge = CvBridge()

        # Depth image storage
        self.latest_depth = None

        # Camera intrinsics
        self.camera_intrinsics = None

        # OpenCV HOG + SVM people detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

    def camera_info_callback(self, msg):
        if self.camera_intrinsics is not None:
            return  # Already set
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
        display_frame = frame.copy()

        # Detect humans in the image
        rects, weights = self.hog.detectMultiScale(frame, winStride=(8,8), padding=(8,8), scale=1.05)

        for i, (x, y, w, h) in enumerate(rects):
            # Draw bounding box on display frame
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Get center of bounding box
            cx = x + w // 2
            cy = y + h // 2

            # Take median depth around the bounding box for robustness
            depth_window = self.latest_depth[max(0,cy-5):cy+5, max(0,cx-5):cx+5]
            if depth_window.size == 0:
                continue
            depth = float(np.median(depth_window))
            if depth == 0.0 or np.isnan(depth):
                continue

            X, Y, Z = self.pixel_to_3d(cx, cy, depth)

            # Publish TF
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "camera_link"
            t.child_frame_id = f"human_{i}"
            t.transform.translation.x = X/1000.0  # Convert mm to meters
            t.transform.translation.y = Y/1000.0
            t.transform.translation.z = Z/1000.0
            t.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(t)

        # Show the frame with detections
        cv2.imshow("Human Detection", display_frame)
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
