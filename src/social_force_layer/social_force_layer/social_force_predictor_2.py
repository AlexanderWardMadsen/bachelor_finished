# Social Force Predictor Node

# Predicts future human positions using Social Force Model.

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
import numpy as np
from social_force_layer.social_force_model import SocialForceModel



class SocialForcePredictor(Node):
    def __init__(self):
        super().__init__('social_force_predictor')
        
        #bedre 
        # Parameters
        self.declare_parameter('frame_id', 'camera_link')
        self.declare_parameter('prediction_time', 5.0) # seconds################
        self.declare_parameter('time_step', 1.0) # time step for simulation (seconds)###########
        
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.prediction_time = float(self.get_parameter('prediction_time').value)
        self.time_step = float(self.get_parameter('time_step').value)
        
        # Calculate number of steps for 5 seconds prediction
        self.prediction_steps = int(self.prediction_time / self.time_step)
        
        self.model = SocialForceModel()

        # Subscribers and Publishers
        # fix
        self.sub_people = self.create_subscription(PoseArray, '/ekf', self.people_callback, 10)
        self.pub_pred = self.create_publisher(PoseArray, '/predicted_humans', 10)
        
        self.get_logger().info(f'Social Force Predictor: predicting {self.prediction_time}s ahead ({self.prediction_steps} steps)')



    # Predict where humans will be in the next 5 seconds
    def people_callback(self, msg: PoseArray) -> None:
        predicted = PoseArray()
        predicted.header.frame_id = self.frame_id
        predicted.header.stamp = self.get_clock().now().to_msg()

        # Extract all people's current positions and velocities
        people = []
        for p in msg.poses:
            pos = np.array([p.position.x, p.position.y])
            vx = p.orientation.x
            vy = p.orientation.y
            vel = np.array([vx, vy])
            
            people.append({'pos': pos, 'vel': vel, 'original_pose': p})

        # Predict each person considering interactions with others
        for i, person in enumerate(people):
            pos = person['pos'].copy()
            vel = person['vel']
            
            # Get positions of all OTHER people for repulsive forces
            others = [people[j]['pos'] for j in range(len(people)) if j != i]
            
            # Simulate prediction_steps into the future with dynamic goal updating
            for _ in range(self.prediction_steps):
                # Update goal dynamically: always X meters ahead in velocity direction
                speed = np.linalg.norm(vel) # find ud af dette 
                if speed > 0.10:
                    goal = pos + vel / speed * 5.0 # 5m ahead from current position
                else:
                    goal = pos # stationary
                
                pos, vel = self.model.step(pos, vel, goal, others, self.time_step)
            
            # Create predicted pose
            pred = Pose()
            pred.position.x = float(pos[0])
            pred.position.y = float(pos[1])
            pred.position.z = 0.0
            pred.orientation = person['original_pose'].orientation # fix
            predicted.poses.append(pred)

        self.pub_pred.publish(predicted)



def main(args=None):
    rclpy.init(args=args)
    node = SocialForcePredictor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
