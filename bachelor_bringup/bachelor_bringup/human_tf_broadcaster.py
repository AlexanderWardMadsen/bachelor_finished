import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped
from openpose import pyopenpose as op  # make sure OpenPose Python API is installed

class HumanTFNode(Node):
    def __init__(self):
        super().__init__('human_tf_node')

        # ROS subscriptions
        self.rgb_sub = self.create_subscription(Image, '/camera/color/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera/color/camera_info', self.camera_info_callback, 10)

        # CV Bridge
        self.bridge = CvBridge()
        self.latest_depth = None
        self.camera_intrinsics = None

        # OpenPose configuration
        params = dict()
        params["model_folder"] = "/home/alex/openpose/models/"  # Set your OpenPose model folder
        params["model_pose"] = "BODY_25"
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()

        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Confidence threshold for keypoints
        self.conf_threshold = 0.3

    def camera_info_callback(self, msg):
        if self.camera_intrinsics is not None:
            return
        self.camera_intrinsics = {
            'fx': msg.k[0],
            'fy': msg.k[4],
            'cx': msg.k[2],
            'cy': msg.k[5]
        }

    def depth_callback(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def rgb_callback(self, msg):
        if self.latest_depth is None or self.camera_intrinsics is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        display_frame = frame.copy()

        datum = op.Datum()
        datum.cvInputData = frame
        self.opWrapper.emplaceAndPop([datum])

        if datum.poseKeypoints is not None:
            for i, person in enumerate(datum.poseKeypoints):
                # Use average of mid-hip and lower torso keypoints (BODY_25 keypoints 8=mid-hip, 11=left hip, 14=right hip)
                keypoints = [8, 11, 14]  # indices in BODY_25
                valid_points = [person[k] for k in keypoints if person[k][2] > self.conf_threshold]
                if not valid_points:
                    continue
                # Average x,y for position
                cx = int(np.mean([p[0] for p in valid_points]))
                cy = int(np.mean([p[1] for p in valid_points]))

                # Median depth around point
                depth_window = self.latest_depth[max(0,cy-5):cy+5, max(0,cx-5):cx+5]
                if depth_window.size == 0:
                    continue
                depth = float(np.median(depth_window))
                if depth == 0.0 or np.isnan(depth):
                    continue

                X, Y, Z = self.pixel_to_3d(cx, cy, depth)

                # Publish TF
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = "camera_link"
                t.child_frame_id = f"human_{i}"
                t.transform.translation.x = X
                t.transform.translation.y = Y
                t.transform.translation.z = Z
                t.transform.rotation.w = 1.0
                self.tf_broadcaster.sendTransform(t)

                # Draw skeleton and keypoints
                for j, kp in enumerate(person):
                    x, y, conf = kp
                    if conf > self.conf_threshold:
                        cv2.circle(display_frame, (int(x), int(y)), 3, (0,255,0), -1)
                cv2.putText(display_frame, f'Person {i}', (cx+10, cy-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        cv2.imshow("OpenPose Human Detection", display_frame)
        cv2.waitKey(1)

    def pixel_to_3d(self, cx, cy, depth):
        fx = self.camera_intrinsics['fx']
        fy = self.camera_intrinsics['fy']
        cx_intr = self.camera_intrinsics['cx']
        cy_intr = self.camera_intrinsics['cy']

        X = (cx - cx_intr) * depth / fx
        Y = (cy - cy_intr) * depth / fy
        Z = depth
        return X, Y, Z

def main(args=None):
    rclpy.init(args=args)
    node = HumanTFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
