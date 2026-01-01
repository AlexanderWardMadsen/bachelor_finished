# ----------------------------------
# this version: 0.1 is compatible with realsense camera but is missing a lot ot the other features
# i also want it to be easy to quicly remove elements or swap them for others though first i need 
# to get a version that is able to transmit to /tf and is able to doble check the deapth values
# with the realsense deapth camera, and make that part modular.
# ----------------------------------
import rclpy
from rclpy.node import Node
import mediapipe as mp
import cv2

from detection.include.camera_manager import CameraManager
from detection.include.tf_plot import TFPlot
from detection.include.skeleton_drawer import SkeletonDrawer
from detection.include.tf_calculator import TFCalculator
from detection.include.pose_model import PoseModel


class PoseAppNode(Node):
    def __init__(self):
        super().__init__('pose_app_node')

        # Declare launch parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('max_people', 4)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        max_people = self.get_parameter('max_people').get_parameter_value().integer_value

        # Use CameraManager to subscribe to camera topics
        self.cam = CameraManager(self)
        self.tf_plot = TFPlot()
        self.pos_x, self.pos_z = {}, {}

        # MediaPipe pose model setup
        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=max_people,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

        # Timer for frame processing loop
        self.timer_period = 1.0 / 30.0  # 30 Hz default
        self.t_ms = 0
        self.ms_per_frame = int(1000 / 30)
        self.timer = self.create_timer(self.timer_period, self.process_frame)

        self.get_logger().info("PoseAppNode initialized and waiting for camera frames...")

    def process_frame(self):
        """Called periodically — processes the latest RGB frame from CameraManager."""
        frame = self.cam.rgb_frame
        if frame is None:
            return  # wait until we have received at least one frame

        h, w = frame.shape[:2]

        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )
        res = self.landmarker.detect_for_video(mp_img, self.t_ms)
        self.t_ms += self.ms_per_frame

        annotated = frame.copy()
        people = res.pose_landmarks or []

        for pid, lm_list in enumerate(people):
            SkeletonDrawer.draw(annotated, lm_list)
            x0, y0, x1, y1 = TFCalculator.bbox(lm_list, w, h)
            cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5

            try:
                l_sh, r_sh = lm_list[11], lm_list[12]
                shoulder_px = abs((l_sh.x - r_sh.x) * w)
            except Exception:
                shoulder_px = max(1.0, x1 - x0)

            est = TFCalculator.estimate_distance(shoulder_px, w)
            if est is None:
                continue

            Z, f_px = est
            X = TFCalculator.pixel_to_world(cx, w, Z, f_px)

            self.pos_x[pid] = TFCalculator.smooth(self.pos_x.get(pid), X)
            self.pos_z[pid] = TFCalculator.smooth(self.pos_z.get(pid), Z)

            hud = f"ID {pid}  X={self.pos_x[pid]:+.2f} m  Z={self.pos_z[pid]:.2f} m"
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (255, 255, 0), 2)
            cv2.circle(annotated, (int(cx), int(cy)), 4, (0, 255, 255), -1)
            cv2.putText(annotated, hud, (x0, max(20, y0 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 220, 255), 2, cv2.LINE_AA)

            self.tf_plot.update_person(pid, float(self.pos_x[pid]), float(self.pos_z[pid]))

        cv2.putText(annotated, f"People: {len(people)}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Skeleton + TF positions", annotated)
        self.tf_plot.step()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info("Quitting PoseAppNode.")
            rclpy.shutdown()

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PoseAppNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
