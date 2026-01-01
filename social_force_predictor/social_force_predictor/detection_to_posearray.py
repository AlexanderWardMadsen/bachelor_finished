#!/usr/bin/env python3
"""
Convert detection Point messages -> PoseArray (/humans_raw)

Listens for Point messages from detector (topic 'depth'):
  point.x = lateral (m)
  point.y = person ID
  point.z = forward (m)

Publishes PoseArray with one Pose per detected person.
"""

from typing import Dict, Tuple

import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseArray, Pose
from std_msgs.msg import Header



class DetectionToPoseArray(Node):
    def __init__(self):
        super().__init__('detection_to_posearray')
        self.declare_parameter('input_point_topic', 'depth')
        self.declare_parameter('output_pose_topic', '/humans_raw')
        self.declare_parameter('publish_interval', 0.5)
        self.declare_parameter('output_frame', 'camera_link')

        self.input_topic = self.get_parameter('input_point_topic').value
        self.output_topic = self.get_parameter('output_pose_topic').value
        self.publish_interval = float(self.get_parameter('publish_interval').value)
        self.output_frame = self.get_parameter('output_frame').value

        # Buffer: person_id -> (x_forward, y_lateral, timestamp)
        self._buffer: Dict[int, Tuple[float, float, float]] = {}

        # ROS interfaces
        self.sub = self.create_subscription(Point, self.input_topic, self._point_cb, 10)
        self.pub = self.create_publisher(PoseArray, self.output_topic, 10)

        self.create_timer(self.publish_interval, self._on_timer)
        self.get_logger().info(f"Detection->PoseArray: listening on '{self.input_topic}', publishing '{self.output_topic}'")



    """Convert Point detection to buffer entry"""
    def _point_cb(self, msg: Point) -> None:
        try:
            pid = int(msg.y)
            x_forward = float(msg.z)
            y_lateral = float(msg.x)
            self._buffer[pid] = (x_forward, y_lateral, time.time())
        except (ValueError, AttributeError):
            pass



    def _on_timer(self) -> None:
        if not self._buffer:
            # publish empty PoseArray with header
            pa = PoseArray()
            pa.header = Header()
            pa.header.stamp = self.get_clock().now().to_msg()
            pa.header.frame_id = self.output_frame
            self.pub.publish(pa)
            return

        pa = PoseArray()
        pa.header = Header()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = self.output_frame

        # One pose per person in buffer
        for person_id, (x_forward, y_lateral, timestamp) in self._buffer.items():
            p = Pose()
            p.position.x = float(x_forward)
            p.position.y = float(y_lateral)
            p.position.z = 0.0
            # Note: timestamp stored for future use (e.g., filtering old detections)
            pa.poses.append(p)

        self.pub.publish(pa)
        self._buffer.clear()



def main(args=None):
    rclpy.init(args=args)
    node = DetectionToPoseArray()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()



if __name__ == '__main__':
    main()
