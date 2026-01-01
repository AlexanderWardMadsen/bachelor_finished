import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped
from detection.include.tf_calculator import TFCalculator# make a comparitor
from detection.include.calculate_z import calculate_z

from geometry_msgs.msg import Point, PointStamped
import rclpy
from rclpy.node import Node

class HumanTFPublisher:
    """Handles depth→3D conversion and TF broadcasting."""

    def __init__(self, node, camera):
        self.node = node
        self.camera = camera
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(node)
        self.publisher_ = self.node.create_publisher(PointStamped, 'depth', 10)


    # ----------------------------------
    # publish the tf of the detected human
    # ----------------------------------
    def publish_pose_tf(self, pixel):
        if self.camera.latest_depth is None or self.camera.camera_intrinsics is None:
            return

        cx, cy = pixel
        depth_window = self.camera.latest_depth[max(0, cy-5):cy+5, max(0, cx-5):cx+5]
        if depth_window.size == 0:
            return
        depth = float(np.median(depth_window))
        if depth == 0.0 or np.isnan(depth):
            return

        X, Y, Z = self.pixel_to_3d(cx, cy, depth)

        t = TransformStamped()
        t.header.stamp = self.node.get_clock().now().to_msg()
        t.header.frame_id = "camera_link"
        t.child_frame_id = "human_0"
        t.transform.translation.x = Z / 1000.0
        t.transform.translation.y = -X / 1000.0
        t.transform.translation.z = -Y / 1000.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.publish(0.0, t.transform.translation.y, t.transform.translation.x)

        self.tf_broadcaster.sendTransform(t)

    # ----------------------------------
    # publish the tf of multiple detected human
    # ----------------------------------
    def publish_pose_tf_array(self, pixel):
        if self.camera.latest_depth is None or self.camera.camera_intrinsics is None:
            return

        for i, (cx, cy) in enumerate(pixel, start=0):
            depth_window = self.camera.latest_depth[max(0, cy-5):cy+5, max(0, cx-5):cx+5]
            if depth_window.size == 0:
                return
            depth = float(np.median(depth_window))
            if depth == 0.0 or np.isnan(depth):
                return

            X, Y, Z = self.pixel_to_3d(cx, cy, depth)

            t = TransformStamped()
            t.header.stamp = self.node.get_clock().now().to_msg()
            t.header.frame_id = "camera_link"
            t.child_frame_id = f"human_{i}"
            # t.transform.translation.x = Z / 1000.0
            # t.transform.translation.y = -X / 1000.0
            # t.transform.translation.z = -Y / 1000.0
            t.transform.translation.x = calculate_z.calculate_true_z(Y, Z) / 1000.0
            t.transform.translation.y = -X / 1000.0
            t.transform.translation.z = 0.0
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0

            self.publish(float(i), t.transform.translation.y, t.transform.translation.x)

            self.tf_broadcaster.sendTransform(t)
    
    # ----------------------------------
    # publish Point message for a person ID
    # ----------------------------------
    def publish(self, pid, x, z):
        point = PointStamped()
        point.header.stamp.sec = self.camera.last_timestamp_sec
        point.header.stamp.nanosec = self.camera.last_timestamp_nsec
        point.point.x = float(x)
        point.point.y = float(pid)
        point.point.z = float(z)
        self.publisher_.publish(point)

    def pixel_to_3d(self, cx, cy, depth):
        fx = self.camera.camera_intrinsics['fx']
        fy = self.camera.camera_intrinsics['fy']
        cx_intr = self.camera.camera_intrinsics['cx']
        cy_intr = self.camera.camera_intrinsics['cy']

        X = (cx - cx_intr) * depth / fx
        Y = (cy - cy_intr) * depth / fy
        Z = depth
        return X, Y, Z
