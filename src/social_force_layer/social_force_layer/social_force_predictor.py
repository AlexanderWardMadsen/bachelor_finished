# imports and such
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose 
import numpy as np
from social_force_layer.social_force_model import SocialForceModel
from typing import Dict, Tuple, List
import csv

# Social Force Predictor Node
class SocialForcePredictor(Node):
    # initialize the predicter class
    def __init__(self):
        super().__init__('social_force_predictor')
        
        # probably not needed
        # Parameters
        self.frame_id = 'camera_link' # probably not needed
        # self.prediction_time = 5.0 # how much time to predict ahead [s]
        self.time_pr_step = 0.5    # time step per prediction step (seconds)
        self.prediction_steps = 5 # make this many predictions 
        # self.prediction_steps = int(self.prediction_time / self.time_pr_step) # make sure this i right 
        self.prediction_time = self.time_pr_step * self.prediction_steps # how much time to predict ahead [s]
        
        self.model = SocialForceModel()

        self.persons: Dict[int, Tuple[float, float, float, float, float]] = {}

        self.flag = True

        # Subscribers and Publishers
        self.sub_people = self.create_subscription(PoseArray, '/ekf', self.people_callback, 10)
        self.pub_pred = self.create_publisher(PoseArray, '/predicted_humans', 10) #find new

        with open("Predictions.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["ID;Weight;X;Z;Time"])


    # make a callback for people data that pulls the data and processes it
    def people_callback(self, msg: PoseArray):
        if not self.flag:
            return
        self.flag = False
        # update persons data
        self.persons.clear()
        # extract people data from PoseArray message
        for idx, pose in enumerate(msg.poses):
            x = pose.position.x      # position data provided in PoseArray
            y = pose.position.y      # position data provided in PoseArray   
            vx = pose.orientation.x  # velocity data provided in PoseArray
            vy = pose.orientation.y  # velocity data provided in PoseArray
            time= pose.orientation.z  # time data provided in PoseArray

            if vx <= 0.1 and vx >= -0.1:
                vx = 0.0
            if vy <= 0.1 and vy >= -0.1:
                vy = 0.0
            self.persons[pose.position.z] = [x, y, vx, vy, time]  # Using position.z as ID
        history = self.predictor()
        self.publish_predictions(history)
        self.flag = True



    # make main predictor 
    def predictor(self) -> List[Dict[int, Dict[str, Tuple[float, float]]]]:
        # make it so that takes the date from persons in NewFilter 
        # make a calculator that calculates goal and other needed medthods
        # make a nested calculation so that all calcs can be made with predicted pathing in mind 

        goals:           Dict[int, Tuple[float, float]]     ={}
        poses:           Dict[int, Tuple[float, float]] ={}
        velocities:      Dict[int, Tuple[float, float]] ={}


        # storage for all steps
        # structure: history[step][id] = {"pose": ..., "velocity": ...}
        history = []
        # current step
        step_data = {}

        for id, person in self.persons.items():
            # self.get_logger().info(f'{person} and {id}')
            poses[id] = np.array(person[0:2])
            velocities[id] = np.array(person[2:4])
            # goal = point[m] + velocity[m/s] * time[s]
            goals[id] = poses[id] + (velocities[id] * self.prediction_time)
            step_data[id] = {
                    "step_id": 0,
                    "pose": poses[id],
                    "velocity": velocities[id],
                    "time": person[4]
                }
        
        history.append(step_data)

        
        # make prediction for all persons
        # first calculate others and then do the prediction step by step
        # make a copy of poses to update them step by step
        for step in range(self.prediction_steps):
            poses_next:      Dict[int, Tuple[float, float]] ={}
            velocities_next: Dict[int, Tuple[float, float]] ={}
            for id in goals.keys():
                # get others
                others = [poses[j] for j in goals.keys() if j != id]
                # do prediction step
                poses_next[id], velocities_next[id] = self.model.step(
                    poses[id], 
                    velocities[id], 
                    goals[id], 
                    others, 
                    self.time_pr_step
                )

            # current step
            step_data = {}  
            # update the final predicted position and velocity
            for id in goals.keys():
                poses[id] = poses_next[id]
                velocities[id] = velocities_next[id]
                step_data[id] = {
                    "step_id": step + 1,
                    "pose": poses_next[id],
                    "velocity": velocities_next[id],
                    "time": 0.0
                }
            
            history.append(step_data)
        
        return history





    # Make a method that publishes the person data in a good way 
    def publish_predictions(self, history: List[Dict[int, Dict[str, Tuple[float, float]]]]):
        pose_array = PoseArray()
        pose_array.header.frame_id = self.frame_id
        pose_array.header.stamp = self.get_clock().now().to_msg()

        for step_idx, step_data in enumerate(history):
            for person_id, data in step_data.items():
                pose = Pose()
                pose.position.x = data["pose"][0]
                pose.position.y = data["pose"][1]
                pose.position.z = 1 - (data["step_id"] / (2*self.prediction_steps))  # encode step in z
                pose.orientation.x = data["velocity"][0]
                pose.orientation.y = data["velocity"][1]
                pose.orientation.z = 0.0
                pose_array.poses.append(pose)
                with open("Predictions.csv", 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([f"{person_id};{1 - (data['step_id'] / (2*self.prediction_steps))};{data['pose'][0]};{data['pose'][1]};{data['time']}"])
        self.pub_pred.publish(pose_array)

        


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
