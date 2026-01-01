import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

class CameraManager:
    """Handles ROS image subscriptions and stores the latest frames and intrinsics."""

    def __init__(self, node):
        self.node = node
        self.bridge = CvBridge()
        self.latest_depth = None
        self.camera_intrinsics = None
        self.rgb_frame = None

        node.create_subscription(Image, '/camera/color/image_raw', self.rgb_callback, 1)
        node.create_subscription(Image, '/camera/depth/image_rect_raw', self.depth_callback, 1)
        node.create_subscription(CameraInfo, '/camera/color/camera_info', self.camera_info_callback, 1)

    def rgb_callback(self, msg):
        self.rgb_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.last_timestamp_sec = msg.header.stamp.sec
        self.last_timestamp_nsec = msg.header.stamp.nanosec

    def depth_callback(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')

    def camera_info_callback(self, msg):
        if self.camera_intrinsics is not None:
            return
        self.camera_intrinsics = {
            'fx': msg.k[0],
            'fy': msg.k[4],
            'cx': msg.k[2],
            'cy': msg.k[5]
        }
        self.node.get_logger().info(f"Camera intrinsics received: {self.camera_intrinsics}")