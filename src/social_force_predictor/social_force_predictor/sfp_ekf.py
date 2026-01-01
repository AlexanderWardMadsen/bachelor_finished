import math
import time
from typing import Dict, List, Tuple

import numpy as np
import rclpy
import csv
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseArray, Pose


# --------------------------------------------------------------------
# class used to make different persons objects to store data
# --------------------------------------------------------------------
class Person:
    # The class needed to store persons data for 1 person
    def __init__(self, time: float, px: float, py: float):
        # State vector: [px, py, vx, vy]
        self.last_state = np.array([px, py, 0.0, 0.0], dtype=float)

        # predicted next position of person
        self.predicted_state = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)

        # stored path of person
        self.stored_path = []  # list to hold stored positions and times
        self.stored_path.append((px, py, 0.0, 0.0, time))

        # Last update time
        self.last_update = time

        self.x = np.copy(self.last_state) # state vector [px, py, vx, vy]
        self.P = np.diag([0.5, 0.5, 1.0, 1.0]) # initial covariance

        # Predicted state and covariance (filled by predict_linear)
        self.x1 = np.copy(self.x)
        self.P1 = np.copy(self.P)

    # method for adding current data to stored path
    def add_to_stored_path(self, time: float, px: float, py: float):
        # add current position and time to stored path
        # and add to current state

        if self.last_update != 0.0:
            dt = time - self.last_update
            if dt > 0:
                vx = (px - self.last_state[0]) / dt
                vy = (py - self.last_state[1]) / dt
                self.stored_path.append((px, py, vx, vy, time))
                self.last_state = np.array([px, py, vx, vy], dtype=float)
        else:
            self.stored_path.append((px, py, 0.0, 0.0, time))
            self.last_state = np.array([px, py, 0.0, 0.0], dtype=float)

    # update predicted position
    def update_predicted_position(self, time: float):
        # make prediction
        if self.last_update != 0.0:
            dt = time - self.last_update
            if dt > 0:
                # simple constant velocity model for prediction
                px = self.last_state[0] + self.last_state[2] * dt
                py = self.last_state[1] + self.last_state[3] * dt
                self.predicted_state = np.array([px, py, self.last_state[2], self.last_state[3]], dtype=float)
        else:
            self.predicted_state = self.last_state.copy()

    def predict_linear(self, time, q_scale):
        dt = time - self.last_update
        # simple constant velocity linear predict for covariance update
        Q = np.eye(4) * (q_scale * dt)
        F = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.x1 = F @ self.x # Predicts the next state.
        self.P1 = F @ self.P @ F.T + Q # Propagates the covariance, adding process noise Q.



class SocialForceEKF(Node):
    def __init__(self):
        super().__init__('sfp_ekf')

        self.gating = 0.75             # association distance threshold [m]
        self.Q_scale = 0.5            # process noise scale 
        self.R_meas = 0.2             # measurement noise stddev
        self.person_timeout = 1.0     # seconds to delete stale persons

        # tracks, id -> Track
        self.persons: Dict[int, Person] = {}
        self._next_id = 1

        # ROS interfaces - subscribe directly to detection Point messages
        self.sub_point = self.create_subscription(
            PointStamped, 'depth', self.depth_callback, 10)
        self.pub = self.create_publisher(PoseArray, '/ekf', 10)
        

        # with open("ekf_stored", 'w', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow(["ID;X;Z;vX;vY;Time"])
        with open("ekf_test", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["ID;X;Z;vX;vY;Time"])



    # make a callback for depth data that pulls the data and processes it
    def depth_callback(self, msg):
        detection = [float(msg.point.z), float(msg.point.x),
                     float(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)]
        self._process_detections(detection)



    def _process_detections(self, detection) -> None:
        # Predict all tracks to current time using simple linear predict
        for person in self.persons.values():
            person.predict_linear(detection[2], self.Q_scale)

        dx = []  # list to hold distances and corresponding person ids
        dx.append([100.0, 0])
        for id, person in self.persons.items():
            # self.get_logger().info(f'{person.x} and {id}')
            pred_pos = person.x1[0:2]
            dist = math.hypot(pred_pos[0] - detection[0], pred_pos[1] - detection[1])
            dx.append([dist, id])

        # dx.sort(key=lambda x: x[0])
        # dist, best_id = dx[0]
        best_dist, best_id = min(dx, key=lambda x: x[0])
        # make new person object if new detection or no detection
        if best_dist < 0.1 and detection[2] == self.persons[best_id].last_update:
            # too close to existing person, ignore
            pass
        elif best_dist <= self.gating:
            # update existing person
            person.add_to_stored_path(detection[2], detection[0], detection[1])
            # update best_id with detection
            self._update_person(self.persons[best_id], detection)
            # self.write_stored_csv(self.persons[best_id], best_id)
            self.write_test_csv(self.persons[best_id], best_id)
        else:
            # create new person
            new_person = Person(detection[2], detection[0], detection[1])
            self.persons[self._next_id] = new_person
            # self.write_stored_csv(self.persons[self._next_id], self._next_id)
            self.write_test_csv(self.persons[self._next_id], self._next_id)
            self._next_id += 1

        # remove stale tracks
        to_delete = []
        for id, person in self.persons.items():
            if detection[2] - person.last_update > self.person_timeout:
               to_delete.append(id)
        for id in to_delete:
            del self.persons[id]

        self.publish()



    def _update_person(self, person: Person, detection) -> None:
        z = np.array([detection[0], detection[1]])
        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        R = np.eye(2) * (self.R_meas ** 2)

        # Measurement prediction
        z_pred = H @ person.x1 # predicted measurement
        S = H @ person.P1 @ H.T + R # innovation covariance
        K = person.P1 @ H.T @ np.linalg.inv(S) # Kalman gain
        y = z - z_pred # measurement residual
        person.x = person.x1 + K @ y # state update
        person.P = (np.eye(4) - K @ H) @ person.P1 # covariance update
        person.last_update = detection[2] # timestamp update
        person.last_state = np.copy(person.x) # sync last_state with updated EKF state



    # def write_stored_csv(self, person: Person, id):
    #     with open("ekf_stored", 'a', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerow([f"{id};{person.stored_path[-1][0]};{person.stored_path[-1][1]};{person.stored_path[-1][2]};{person.stored_path[-1][3]};{person.last_update}"])


    def write_test_csv(self, person: Person, id):
        with open("ekf_test", 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f"{id};{person.last_state[0]};{person.last_state[1]};{person.last_state[2]};{person.last_state[3]};{person.last_update}"])

    def publish(self):
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "map"

        for id, person in self.persons.items():
            pose = Pose()
            pose.position.x = person.last_state[0]
            pose.position.y = person.last_state[1]
            pose.position.z = float(id)
            pose.orientation.x = person.last_state[2]
            pose.orientation.y = person.last_state[3]
            pose.orientation.z = person.last_update
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)

        self.pub.publish(pose_array)


def main(args=None):
    rclpy.init(args=args)
    node = SocialForceEKF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
