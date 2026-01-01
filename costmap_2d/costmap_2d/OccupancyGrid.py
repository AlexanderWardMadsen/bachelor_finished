"""
costmap_2d node

Simple ROS2 node that listens for a `PoseArray` on `/humans` and publishes
an `OccupancyGrid` marking occupied cells around human positions. The node
is parameterized (resolution, width/height, inflation radius, frame_id).
"""

from typing import List, Tuple
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from nav_msgs.msg import OccupancyGrid, MapMetaData
from std_msgs.msg import Header
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped


def _quat_to_rot_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Convert quaternion (x,y,z,w) to a 3x3 rotation matrix."""
    # normalize
    nq = x*x + y*y + z*z + w*w
    if nq < 1e-12:
        return np.eye(3)
    s = 2.0 / nq
    xx = x * x * s
    yy = y * y * s
    zz = z * z * s
    xy = x * y * s
    xz = x * z * s
    yz = y * z * s
    wx = w * x * s
    wy = w * y * s
    wz = w * z * s
    R = np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ], dtype=float)
    return R



class Costmap2D(Node):
    def __init__(self):
        super().__init__('costmap_2d')

        # Parameters (can be overridden via ros2 param set or launch)
        self.declare_parameter('resolution', 0.1) # meters / cell
        self.declare_parameter('width', 200) # number of cells
        self.declare_parameter('height', 200)
        self.declare_parameter('origin_x', -10.0) # world coordinates of cell (0,0)
        self.declare_parameter('origin_y', -10.0)
        self.declare_parameter('frame_id', 'camera_link')
        self.declare_parameter('inflation_radius', 0.5) # meters to inflate around humans
        self.declare_parameter('occupied_value', 100)
        self.declare_parameter('free_value', 0)
        self.declare_parameter('publish_rate', 1.0) # Hz

        self.resolution = float(self.get_parameter('resolution').value)
        self.width = int(self.get_parameter('width').value)
        self.height = int(self.get_parameter('height').value)
        self.origin_x = float(self.get_parameter('origin_x').value)
        self.origin_y = float(self.get_parameter('origin_y').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.inflation_radius = float(self.get_parameter('inflation_radius').value)
        self.occupied_value = int(self.get_parameter('occupied_value').value)
        self.free_value = int(self.get_parameter('free_value').value)
        self.publish_rate = float(self.get_parameter('publish_rate').value)

        # Internal storage of last seen human poses
        self.humans: List[Tuple[float, float]] = []

        # ROS interfaces
        self.sub_people = self.create_subscription(PoseArray, '/predicted_humans', self._people_callback, 10)
        self.pub_grid = self.create_publisher(OccupancyGrid, '/social_obstacles', 10)

        # TF buffer/listener to transform incoming PoseArray into costmap frame
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Timer to publish grid at a steady rate
        self.create_timer(1.0 / max(self.publish_rate, 1e-6), self._timer_cb)

        self.get_logger().info(f'Costmap2D node started: {self.width}x{self.height} @ {self.resolution} m')

    # -------------------- callbacks --------------------
    def _people_callback(self, msg: PoseArray) -> None:
        # Extract x,y positions from incoming PoseArray and store them
        positions: List[Tuple[float, float]] = []

        # If the incoming PoseArray is in a different frame, try to transform
        src_frame = msg.header.frame_id if hasattr(msg.header, 'frame_id') else ''
        if src_frame and src_frame != self.frame_id:
            try:
                t: TransformStamped = self.tf_buffer.lookup_transform(self.frame_id, src_frame, rclpy.time.Time())
                # build rotation matrix from quaternion
                q = t.transform.rotation
                R = _quat_to_rot_matrix(q.x, q.y, q.z, q.w)
                trans = np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z], dtype=float)
                for p in msg.poses:
                    src = np.array([float(p.position.x), float(p.position.y), float(p.position.z)], dtype=float)
                    tgt = R @ src + trans
                    positions.append((float(tgt[0]), float(tgt[1])))
            except Exception as e:
                # If transform failed, fallback to using the raw positions but log a warning
                self.get_logger().warning(f"Failed to transform PoseArray from '{src_frame}' to '{self.frame_id}': {e}. Using raw positions.")
                for p in msg.poses:
                    positions.append((float(p.position.x), float(p.position.y)))
        else:
            for p in msg.poses:
                positions.append((float(p.position.x), float(p.position.y)))

        self.humans = positions

    def _timer_cb(self) -> None:
        grid = self._build_grid()
        self._publish_grid(grid)

    # -------------------- grid helpers --------------------
    def _build_grid(self) -> np.ndarray:
        # Initialize grid with free values
        grid = np.full((self.height, self.width), self.free_value, dtype=np.int8)

        if not self.humans:
            return grid

        # Precompute inflation in cells
        inflation_cells = int(math.ceil(self.inflation_radius / self.resolution))

        for (hx, hy) in self.humans:
            gx, gy = self.world_to_grid(hx, hy)
            if gx is None:
                continue
            # Inflate a square around the human (approximate circle)
            x0 = max(0, gx - inflation_cells)
            x1 = min(self.width - 1, gx + inflation_cells)
            y0 = max(0, gy - inflation_cells)
            y1 = min(self.height - 1, gy + inflation_cells)

            for yy in range(y0, y1 + 1):
                for xx in range(x0, x1 + 1):
                    dx = (xx + 0.5) * self.resolution + self.origin_x - hx
                    dy = (yy + 0.5) * self.resolution + self.origin_y - hy
                    dist = math.hypot(dx, dy)
                    if dist <= self.inflation_radius:
                        grid[yy, xx] = self.occupied_value

        return grid

    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int] | Tuple[None, None]:
        # Convert world coordinates to grid indices
        gx = int(math.floor((wx - self.origin_x) / self.resolution))
        gy = int(math.floor((wy - self.origin_y) / self.resolution))
        if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height:
            return (None, None)
        return (gx, gy)

    def _publish_grid(self, grid: np.ndarray) -> None:
        grid_msg = OccupancyGrid()
        grid_msg.header = Header()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = self.frame_id

        info = MapMetaData()
        info.resolution = float(self.resolution)
        info.width = int(self.width)
        info.height = int(self.height)
        info.origin.position.x = float(self.origin_x)
        info.origin.position.y = float(self.origin_y)
        info.origin.position.z = 0.0
        info.origin.orientation.x = 0.0
        info.origin.orientation.y = 0.0
        info.origin.orientation.z = 0.0
        info.origin.orientation.w = 1.0
        grid_msg.info = info

        # OccupancyGrid.data is a flat list in row-major order (y then x).
        grid_msg.data = grid.flatten(order='C').astype(np.int8).tolist()

        self.pub_grid.publish(grid_msg)


def main(args=None):
    rclpy.init(args=args)
    node = Costmap2D()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
