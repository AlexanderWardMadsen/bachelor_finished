import math
from typing import Dict

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseArray, PointStamped

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
        self.last_update = time

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
        
        
        


# --------------------------------------------------------------------
# filter ROS Node
# --------------------------------------------------------------------
class NewFilter(Node):
    def __init__(self):
        super().__init__('new_filter')

        # ---------------- Parameters ----------------
        # Load parameters
        self.gating = 1.0               # max association distance
        # self.Q_scale = 0.5              # process noise scale
        # self.R_meas = 0.2               # measurement noise stddev
        self.person_death = 5.0         # seconds to delete stale tracks
        self.publish_rate = 15.0        # Hz to publish predictions
        # self.prediction_horizon = 10    # steps to predict ahead 

        # # Social Force Model
        # self.sfm = SocialForceModel()

        # Track database
        self.persons: Dict[int, Person] = {}
        self._next_id = 1

        # ---------------- ROS Interfaces ----------------
        self.subscription = self.create_subscription(
            PointStamped, "depth", self.depth_callback, 10)

    # make a callback for depth data that pulls the data and processes it
    def depth_callback(self, msg):
        detection = [float(msg.point.z), float(msg.point.x),
                     float(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)]
        
        self._sort_detections(detection)
    
    # make a genaral nearerest neighbor calculator 


    # processing detection data what existing persons we have and updating them or creating new ones
    def _sort_detections(self, detections):
        candidates = []
        # update new predicted position
        for person in self.persons.values():
            person.update_predicted_position(detections[2])

        # compare nearest neighbor to find the best match for existing persons
        for id, person in self.persons.items():
            self.get_logger().info(f'{person.stored_path} and {id}')
            # compare time stamps
            if person.last_update == detections[2]:
                continue
            # compare positions
            # compare velocities
            # compare in x and y direction for added accuracy
            candidates.append({
                "id":   id, 
                # "x_direction": (detections[0] - person.predicted_state[0]) * person.predicted_state[2] > 0,
                # "y_direction": (detections[1] - person.predicted_state[1]) * person.predicted_state[3] > 0,
                "x_distance_pred":  (detections[0] - person.predicted_state[0]),
                "y_distance_pred":  (detections[1] - person.predicted_state[1]),
                "x_distance_last":  (detections[0] - person.last_state[0]),
                "y_distance_last":  (detections[1] - person.last_state[1]),
                "x_velocity":       (detections[0] - person.last_state[0]) / (detections[2] - person.last_update),
                "y_velocity":       (detections[1] - person.last_state[1]) / (detections[2] - person.last_update)
            })

        best_candidate = [None, None]
        temp_best_candidate = [None, None]
        for candidate in candidates:
            hyp_dist_pred = math.hypot(candidate["x_distance_pred"], candidate["y_distance_pred"])
            hyp_dist_last = math.hypot(candidate["x_distance_last"], candidate["y_distance_last"])
            # set some thresholds for matching
            if hyp_dist_pred < hyp_dist_last:
                if hyp_dist_pred < self.gating:
                    continue
                temp_best_candidate = [candidate["id"], hyp_dist_pred]
            else:
                if hyp_dist_last < self.gating:
                    continue
                elif (abs(self.persons[candidate["id"]].last_state[2]) <= 0.3 and
                      abs(self.persons[candidate["id"]].last_state[3]) <= 0.3 and
                      abs(self.persons[candidate["id"]].last_state[2]) >= -0.3 and
                      abs(self.persons[candidate["id"]].last_state[3]) >= -0.3):
                    best_candidate = [candidate["id"], 0.0]
                    break
                
                temp_best_candidate = [candidate["id"], hyp_dist_last]

            if (candidate["x_velocity"] > self.persons[candidate["id"]].last_state[2]-0.2 and 
                candidate["x_velocity"] < self.persons[candidate["id"]].last_state[2]+0.2):
                temp_best_candidate[1] *= 0.8
            if (candidate["y_velocity"] > self.persons[candidate["id"]].last_state[3]-0.2 and 
                candidate["y_velocity"] < self.persons[candidate["id"]].last_state[3]+0.2):
                temp_best_candidate[1] *= 0.8

            #update best candidate
            if temp_best_candidate[1] < best_candidate[1] or temp_best_candidate[0] is not None:
                self.get_logger().info(f'filter: {temp_best_candidate[0]} m')
                best_candidate = temp_best_candidate.copy()

        # make new person object if new detection or no detection
        if best_candidate[0] is not None:
            # update existing person
            person = self.persons[best_candidate[0]]
            person.add_to_stored_path(detections[2], detections[0], detections[1])
        else:
            # create new person
            new_person = Person(detections[2], detections[0], detections[1])
            self.persons[self._next_id] = new_person
            self._next_id += 1
        

        # # delete old persons if no detection for a while
        # for id, person in self.persons.items():
        #     if detections[2] - person.last_update >= self.person_death:
        #         del self.persons[id]

        to_delete = []
        for id, person in self.persons.items():
            if detections[2] - person.last_update >= self.person_death:
                to_delete.append(id)
        for id in to_delete:
            del self.persons[id]


# --------------------------------------------------------------------
# Main function
# --------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = NewFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()