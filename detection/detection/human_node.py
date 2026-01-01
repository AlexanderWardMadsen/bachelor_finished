import rclpy
from rclpy.node import Node
import cv2

from detection.include.camera_manager import CameraManager
from detection.include.pose_detector import PoseDetector
from detection.include.tf_publisher import HumanTFPublisher

class HumanNode(Node):
    """Main ROS2 node tying together camera, pose detection, and TF broadcasting."""

    def __init__(self):
        super().__init__('human_tf_node')
        self.camera = CameraManager(self)
        self.detector = PoseDetector()
        self.publisher = HumanTFPublisher(self, self.camera)

        # Timer to process frames at ~30 Hz
        self.timer = self.create_timer(1.0 / 30.0, self.process_frame)

    def process_frame(self):
        if self.camera.rgb_frame is None:
            return

        frame = self.camera.rgb_frame.copy()
        pixel, conf = self.detector.detect_nose(frame)

        if pixel:
            self.publisher.publish_pose_tf(pixel)
            cv2.circle(frame, pixel, 5, (0, 255, 0), -1)
            cv2.putText(frame, f'Conf: {conf:.2f}', (pixel[0] + 10, pixel[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Human Detection", frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = HumanNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
