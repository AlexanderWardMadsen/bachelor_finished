# ros2 run bachelor_bringup ekf_test_logger /<topic> <name>.csv

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
import csv
from datetime import datetime

class PointLogger(Node):
    def __init__(self, topic_name, file_name):
        super().__init__('point_logger')
        self.subscription = self.create_subscription(
            PointStamped,
            topic_name,
            self.point_callback,
            10
        )
        self.file_name = file_name

        # Create file and write header
        with open(self.file_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["ID;X;Z;Time"])

        self.get_logger().info(f"Logging {topic_name} to {file_name}")

    def point_callback(self, msg):
        y_val = msg.point.y   # This is now the first value
        x_val = msg.point.x
        z_val = msg.point.z
        utc_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        with open(self.file_name, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([f"{y_val};{x_val};{z_val};{utc_time}"])

        self.get_logger().info(f"Logged: Y={y_val}, X={x_val}, Z={z_val}, UTC={utc_time}")

    def write_five_columns_csv(self, data):
        with open(self.file_name, "w", newline="") as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(data)


def main(args=None):
    rclpy.init(args=args)

    # Retrieve topic name and filename from launch arguments
    import sys
    if len(sys.argv) != 3:
        print("Usage: ros2 run <package> <node> <topic_name> <file_name>")
        return

    topic_name = sys.argv[1]
    file_name = sys.argv[2]

    point_logger = PointLogger(topic_name, file_name)

    try:
        rclpy.spin(point_logger)
    except KeyboardInterrupt:
        pass

    point_logger.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
