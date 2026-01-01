"""
Visualize costmap from costmap_2d with blue-to-red color scheme.

Subscribes to OccupancyGrid and publishes colored Image for visualization.
Blue = low cost, Red = high cost.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
import cv2



class CostmapVisualizer(Node):
    def __init__(self):
        super().__init__('costmap_visualizer')

        self.declare_parameter('costmap_topic', '/social_obstacles')
        self.declare_parameter('image_topic', '/costmap_visualization')

        self.costmap_topic = str(self.get_parameter('costmap_topic').value)
        self.image_topic = str(self.get_parameter('image_topic').value)

        self.sub = self.create_subscription(OccupancyGrid, self.costmap_topic, self._costmap_cb, 10)
        self.pub = self.create_publisher(Image, self.image_topic, 10)

        self.get_logger().info(f'Visualizing {self.costmap_topic} -> {self.image_topic}')



    """Convert OccupancyGrid costmap to colored image (blue=low, red=high)"""
    def _costmap_cb(self, msg: OccupancyGrid) -> None:
        try:
            # Extract cost data
            height = msg.info.height
            width = msg.info.width
            data = np.array(msg.data, dtype=np.int8)
            
            # Reshape to 2D grid
            grid = data.reshape((height, width))
            
            # Convert to uint8 (0-255 range)
            # Handle special values: -1=unknown, 254=lethal, 255=unknown
            # Treat unknown (-1) as free (0)
            grid_u8 = np.where(grid < 0, 0, grid).astype(np.uint8)
            
            # Apply JET colormap (blue=low, red=high)
            colored = cv2.applyColorMap(grid_u8, cv2.COLORMAP_JET)
            # Convert BGR to RGB
            rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            
            # Create and publish Image message
            img_msg = Image()
            img_msg.header.stamp = msg.header.stamp
            img_msg.header.frame_id = 'camera_link'
            img_msg.height = height
            img_msg.width = width
            img_msg.encoding = 'rgb8'
            img_msg.is_bigendian = False
            img_msg.step = width * 3
            img_msg.data = rgb.tobytes()
            
            self.pub.publish(img_msg)
            
        except Exception as e:
            self.get_logger().error(f'Failed to visualize costmap: {e}')



def main(args=None):
    rclpy.init(args=args)
    node = CostmapVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
