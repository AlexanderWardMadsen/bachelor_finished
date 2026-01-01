from typing import List, Tuple, Dict
import math
from scipy.stats import multivariate_normal
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import OccupancyGrid
import numpy as np
from social_force_layer.social_force_predictor import SocialForcePredictor



class Costmap2D(Node):
    def __init__(self):
        super().__init__('costmap_2d')

        # grid parameters
        self.resolution = 0.1   # meters / cell
        self.width = 200        # number of cells
        self.height = 200
        self.origin_x = 0.0     # world coordinates of cell (0,0)
        self.origin_y = 0.0
        # self.frame_id = 'camera_link'
        self.free_value = 0     # check
        self.cost_max = 100     # all the above are special cases 252
        self.publish_rate = 30.0 # Hz

        # human cost parameters
        self.inflation_radius = 0.75  # meters to inflate around humans
        self.weight_front = 0.5
        self.weight_back = 1.5
        self.weight_side = 0.5
        self.human_amplitude = 0.35
        self.lethal_threshold = 0.7


        # Initialice predictor 
        self.humans = []

        # ROS interfaces
        self.sub_people = self.create_subscription(PoseArray, '/predicted_humans', self._people_callback, 10)
        self.pub_grid = self.create_publisher(OccupancyGrid, '/social_obstacles', 10)

        self.is_busy = False
        # self.timer = self.create_timer(1.0 / self.publish_rate, self.cykle_runner)


    # make a callback for people data that pulls the data and processes it
    def _people_callback(self, msg: PoseArray):
        # update persons data
        self.humans.clear()
        # extract people data from PoseArray message
        for idx, pose in enumerate(msg.poses):
            x = pose.position.x      # position data provided in PoseArray
            y = pose.position.y      # position data provided in PoseArray   
            weight = pose.position.z  # weight data provided in PoseArray
            vx = pose.orientation.x  # velocity data provided in PoseArray
            vy = pose.orientation.y  # velocity data provided in PoseArray

            yaw = math.atan2(vy, vx) #radiance
            
            self.humans.append({'x': x, 'y': y, 'yaw': yaw, 'weight': weight})
        self.cykle_runner()

    # make a method to call and extract data from social_force_predicrtor 
    def get_predictions(self):
        step_idx_weight = 0
        weight = 1
        self.humans = []
        history: List[Dict[int, Dict[str, Tuple[float, float]]]] = self.predictions.predictor()
        for step_idx, step_data in enumerate(history):
            for person_id, data in step_data.items():
                # Extract pose and velocity vectors
                # pose = data["pose"]         # (x, y)
                # velocity = data["velocity"] # (vx, vy)

                # Calculate the weight 
                if step_idx != step_idx_weight:
                    weight -= 1/self.predictions.prediction_steps
                    step_idx_weight += 1
                

                # Unpack them into individual variables
                # x, y = pose
                # vx, vy = velocity
                x, y = data["pose"]
                vx, vy = data["velocity"]
                yaw = math.atan2(vy, vx) #radiance
                self.humans.append({'x': x, 'y': y, 'yaw': yaw, 'weight': weight})

    # make a method that calculates the gausiean and inerts in the grid 
    def build_grid(self):
        # Initialize grid to free
        grid = np.zeros((self.height, self.width), dtype=np.float32)

        # Define the grid coordinates in world frame
        x_lin = np.linspace(self.origin_x, self.origin_x + (self.width-1)* self.resolution, self.width)
        y_lin = np.linspace(self.origin_y, self.origin_y + (self.height-1) * self.resolution, self.height)
        xv, yv = np.meshgrid(x_lin, y_lin)

        # Flatten for vectorized computation
        coords = np.stack([xv, yv], axis=-1)

        # Add human influence
        for human in self.humans:
            hx, hy, yaw, weight = human['x'], human['y'], human['yaw'], human['weight']

            # Covariance matrix for Gaussian (inflation)
            # elongated along human's direction
            cov_x = (self.inflation_radius * self.weight_front)**2
            cov_y = (self.inflation_radius * self.weight_side)**2
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            cov = R @ np.diag([cov_x, cov_y]) @ R.T

            rv = multivariate_normal(mean=[hx+10, hy+10], cov=cov)
            grid = np.maximum(grid, weight * rv.pdf(coords))

            # grid += weight * rv.pdf(coords)

        # Normalize grid to 0..1
        grid = grid / np.max(grid) if np.max(grid) > 0 else grid

        # Map to 0..252
        raw = np.round(grid * float(self.cost_max)).astype(np.int8)
        # raw[raw > self.lethal_threshold * self.cost_max] = 100  # lethal

        return raw  

    # make the timing system that regulates the process 
    def cykle_runner(self):
        if self.is_busy:
            return  # skip if previous cycle not finished
        self.is_busy = True

        # self.get_predictions()
        raw = self.build_grid()
        self.publish_grid(raw)

        self.is_busy = False

    # make a method that transformes the poses 

    # publish costmap
    def publish_grid(self, raw) -> None:
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg() #fix
        grid_msg.header.frame_id = 'camera_link'

        grid_msg.info.resolution = float(self.resolution)
        grid_msg.info.width = int(self.width)
        grid_msg.info.height = int(self.height)
        # grid real world (x, y, rad) of the map's bottom-left corner of map frame in meters
        grid_msg.info.origin.position.x = -10.0 
        grid_msg.info.origin.position.y = -10.0
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation.x = 0.0
        grid_msg.info.origin.orientation.y = 0.0
        grid_msg.info.origin.orientation.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0

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
