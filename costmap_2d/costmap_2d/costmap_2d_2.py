# ------------------------------------------------------------------------------
# costmap_2d node
# Simple ROS2 node that generates a 2D costmap with Gaussian cost functions
# around human positions. Subscribes to human positions and publishes an costmap.
# ------------------------------------------------------------------------------

from typing import List, Tuple
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import OccupancyGrid, MapMetaData
from std_msgs.msg import Header
import numpy as np
from math import atan2



class Costmap2D(Node):
    def __init__(self):
        super().__init__('costmap_2d')

        # grid parameters
        self.resolution = 0.1   # meters / cell
        self.width = 200        # number of cells
        self.height = 200
        self.origin_x = -10.0     # world coordinates of cell (0,0)
        self.origin_y = -10.0
        self.frame_id = 'camera_link'
        self.free_value = 0     # check
        self.cost_max = 255
        self.publish_rate = 1.0 # Hz

        # human cost parameters
        self.inflation_radius = 0.75  # meters to inflate around humans
        self.weight_front = 0.5
        self.weight_back = 1.5
        self.weight_side = 0.5
        self.human_amplitude = 0.35
        self.lethal_threshold = 0.7

        # default weight for predictions without weight field (current detections = 1.0)
        self.default_weight = 1.0

        # ROS interfaces
        self.sub_people = self.create_subscription(PoseArray, '/predicted_humans', self._people_callback, 10)
        self.pub_grid = self.create_publisher(OccupancyGrid, '/social_obstacles', 10)

        # Store human positions
        self.humans: List[dict] = [] # fix 

        # Timer to publish grid at a steady rate
        self.create_timer(1.0 / max(self.publish_rate, 1e-6), self._timer_cb)
        self.get_logger().info(f'Costmap2D node started: {self.width}x{self.height} @ {self.resolution} m')



    """Extract human positions, orientations, and weights from PoseArray"""
    def _people_callback(self, msg: PoseArray) -> None:
        self.humans = []
        if not msg.poses:
            return

        for p in msg.poses:
            px = float(p.position.x)
            py = float(p.position.y)
            # Use position.z as weight if available (predictor encodes it there)
            # Otherwise use default weight for current detections

            pz = getattr(p.position, 'z', None) #make sure this need to exist
            weight = float(pz) if pz is not None and pz > 0 else self.default_weight

            # Yaw from velocity vector (encoded in orientation by predictor/EKF)
            vx = getattr(p.orientation, 'x', 0.0)
            vy = getattr(p.orientation, 'y', 0.0)
            yaw = atan2(vy, vx)  # Direction of motion from velocity vector

            self.humans.append({'x': px, 'y': py, 'yaw': yaw, 'weight': weight})



    """Build costmap grid with Gaussian cost functions around humans"""
    def _build_grid(self) -> np.ndarray:
        grid = np.full((self.height, self.width), float(self.free_value), dtype=float)

        if not self.humans:
            return grid.astype(np.int8)

        inflation_cells = int(math.ceil(self.inflation_radius / self.resolution))

        # Add oriented Gaussian cost potentials per human
        for h in self.humans:
            hx = float(h['x'])
            hy = float(h['y'])
            yaw = float(h['yaw'])
            weight = float(h.get('weight', self.default_weight))  # Prediction confidence/temporal weight Dosn't exist

            gx, gy = self.world_to_grid(hx, hy)

            # is this really needed
            if gx is None:
                cx = int(math.floor((hx - self.origin_x) / self.resolution))
                cy = int(math.floor((hy - self.origin_y) / self.resolution))
                gx, gy = cx, cy

            # Inflate a square around the human
            x0 = max(0, gx - inflation_cells)
            x1 = min(self.width - 1, gx + inflation_cells)
            y0 = max(0, gy - inflation_cells)
            y1 = min(self.height - 1, gy + inflation_cells)

            #check the math here
            # Rotation for orientation
            cos_y = math.cos(-yaw)
            sin_y = math.sin(-yaw)

            # is the +1 needed and fix math 
            for yy in range(y0, y1 + 1):
                for xx in range(x0, x1 + 1):
                    cx = (xx + 0.5) * self.resolution + self.origin_x
                    cy = (yy + 0.5) * self.resolution + self.origin_y
                    dx = cx - hx
                    dy = cy - hy
                    
                    # Rotate into human frame
                    xp = cos_y * dx - sin_y * dy
                    yp = sin_y * dx + cos_y * dy

                    # Directional weights (front vs back)
                    if xp >= 0:
                        sx = max(self.weight_front, 1e-3)
                    else:
                        sx = max(self.weight_back, 1e-3)
                    sy = max(self.weight_side, 1e-3)

                    # Gaussian potential with weight decay (lower weight = lower cost)
                    val = self.human_amplitude * math.exp(-0.5 * ((xp * xp) / (sx * sx) + (yp * yp) / (sy * sy)))
                    cost = float(self.free_value) + val * (float(self.cost_max - self.free_value)) * weight
                    
                    # Take maximum cost at each cell (conservative planning)
                    grid[yy, xx] = max(grid[yy, xx], cost)
        
        grid = np.clip(grid, float(self.free_value), float(self.cost_max))
        return grid.astype(np.int8)



    """Timer callback to publish costmap"""
    def _timer_cb(self) -> None:
        grid = self._build_grid()
        self._publish_grid(grid)



    # Fix
    #  Convert world coordinates to grid indices
    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int] | Tuple[None, None]:
        gx = int(math.floor((wx - self.origin_x) / self.resolution))
        gy = int(math.floor((wy - self.origin_y) / self.resolution))
        if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height:
            return (None, None)
        return (gx, gy)



    """Publish costmap as OccupancyGrid message"""
    def _publish_grid(self, grid: np.ndarray) -> None:
        grid_msg = OccupancyGrid()
        grid_msg.header = Header()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = self.frame_id

        info = MapMetaData()
        info.resolution = float(self.resolution)
        info.width = int(self.width)
        info.height = int(self.height)
        info.origin.position.x = -10.0 # grid real worled (x, y, rad) of the map's bottom-left corner in the map frame
        info.origin.position.y = -10.0
        info.origin.position.z = 0.0
        info.origin.orientation.x = 0.0
        info.origin.orientation.y = 0.0
        info.origin.orientation.z = 0.0
        info.origin.orientation.w = 1.0
        grid_msg.info = info

        # Normalize and convert to cost bytes
        denom = max(1.0, float(self.cost_max - self.free_value))
        norm = (grid - float(self.free_value)) / denom
        norm = np.clip(norm, 0.0, 1.0)

        # Map to byte range: 0..252 graded, 254 lethal, 255 unknown
        usable_max = 252
        raw = np.round(norm * float(usable_max)).astype(np.uint8)
        lethal_mask = norm >= float(self.lethal_threshold)
        raw[lethal_mask] = np.uint8(254)
        grid_msg.data = raw.astype(np.int8).flatten(order='C').tolist()

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
