# ----------------------------------
# this version: 0.2
# ----------------------------------
import rclpy
from rclpy.node import Node
import mediapipe as mp
import cv2

from detection.include.camera_manager import CameraManager
from detection.include.tf_plot import TFPlot
from detection.include.skeleton_drawer import SkeletonDrawer
from detection.include.tf_calculator import TFCalculator
from detection.include.tf_publisher import HumanTFPublisher


class PoseAppNode(Node):
    def __init__(self):
        super().__init__('pose_app_node')

        # Declare launch parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('max_people', 4)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        max_people = self.get_parameter('max_people').get_parameter_value().integer_value

        # Camera manager and plotting
        self.camera = CameraManager(self)
        self.tf_plot = TFPlot(self)
        self.pos_x, self.pos_z = {}, {}

        # TF Publisher
        self.tf_publisher = HumanTFPublisher(self, self.camera)

        # MediaPipe Pose setup
        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=max_people,
            min_pose_detection_confidence=0.8,  # changed from 0.5
            min_pose_presence_confidence=0.8,   # changed from 0.5
            min_tracking_confidence=0.8,        # changed from 0.5
            output_segmentation_masks=False,    # changed from False # faster if False
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

        # Timer for frame processing loop
        self.timer_period = 1.0 / 30.0  # 30 Hz default
        self.t_ms = 0
        self.ms_per_frame = int(1000 / 30)
        self.timer = self.create_timer(self.timer_period, self.process_frame)

        self.get_logger().info("PoseAppNode initialized and waiting for camera frames...")

    def process_frame(self):
        frame = self.camera.rgb_frame
        if frame is None:
            return  # wait until we have received at least one frame

        h, w = frame.shape[:2] # get frame dimensions

        # ----- Run MediaPipe Pose Landmarker -----
        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )
        res = self.landmarker.detect_for_video(mp_img, self.t_ms)
        self.t_ms += self.ms_per_frame

        annotated = frame.copy()
        people = res.pose_landmarks or []

        # Collect pixel positions for TF publishing
        tf_pixel_positions = []

        # ----- Process each detected person -----
        for pid, lm_list in enumerate(people):
            SkeletonDrawer.draw(annotated, lm_list)
            x0, y0, x1, y1 = TFCalculator.bbox(lm_list, w, h)
            cx, cy = int((x0 + x1) * 0.5), int((y0 + y1) * 0.5) #Not an optimal method but works for now
            tf_pixel_positions.append((cx, cy))

            try:
                l_sh, r_sh = lm_list[11], lm_list[12]
                shoulder_px = abs((l_sh.x - r_sh.x) * w)
                shoulder_py = abs((l_sh.y - r_sh.y) * h)
            except Exception:
                shoulder_px = max(1.0, x1 - x0)

            est = TFCalculator.estimate_distance(shoulder_px, w, shoulder_py, h, cy)
            if est is None:
                continue

            Z, f_px = est
            X = TFCalculator.pixel_to_world(cx, w, Z, f_px)

            self.pos_x[pid] = TFCalculator.smooth(self.pos_x.get(pid), X)
            self.pos_z[pid] = TFCalculator.smooth(self.pos_z.get(pid), Z)
            

            hud = f"ID {pid}  X={self.pos_x[pid]:+.2f} m  Z={self.pos_z[pid]:.2f} m"
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (255, 255, 0), 2)
            cv2.circle(annotated, (cx, cy), 4, (0, 255, 255), -1)
            cv2.putText(annotated, hud, (x0, max(20, y0 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 220, 255), 2, cv2.LINE_AA)

            self.tf_plot.update_person(pid, -float(self.pos_x[pid]), float(self.pos_z[pid]))

        # ----- Publish TFs for all humans -----
        if tf_pixel_positions:
            self.tf_publisher.publish_pose_tf_array(tf_pixel_positions)

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
