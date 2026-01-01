# ----------------------------------
# this version: 0.1 
# ----------------------------------

import rclpy
from rclpy.node import Node
import mediapipe as mp
import cv2

# Import local helper modules for camera input, visualization, and calculations
from detection.include.camera_manager import CameraManager
from detection.include.tf_plot import TFPlot
from detection.include.skeleton_drawer import SkeletonDrawer
from detection.include.tf_calculator import TFCalculator
# from detection.include.pose_model import PoseModel  # Optional, not currently used


# ============================================================
# CLASS: PoseAppNode
# A ROS2 node that:
#   - Captures frames from a camera (via CameraManager)
#   - Uses MediaPipe Pose to detect human poses
#   - Estimates 3D position (X, Z) of detected people
#   - Publishes transforms (TFs) and visualizes results
# ============================================================
class PoseAppNode(Node):
    def __init__(self):
        # Initialize the ROS2 node with the name 'pose_app_node'
        super().__init__('pose_app_node')

        # -----------------------------
        # Declare configurable parameters for launch files
        # -----------------------------
        # 'model_path' = path to the MediaPipe pose model file
        # 'max_people' = maximum number of people to detect per frame
        self.declare_parameter('model_path', '')
        self.declare_parameter('max_people', 4)

        # Retrieve the parameter values (provided via ROS2 launch file or CLI)
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        max_people = self.get_parameter('max_people').get_parameter_value().integer_value

        # -----------------------------
        # Setup subsystems
        # -----------------------------
        # CameraManager subscribes to a ROS camera topic and provides RGB frames
        self.camera = CameraManager(self)
        # TFPlot displays a live 2D top-down view of detected human positions
        self.tf_plot = TFPlot()
        # Store smoothed X/Z position estimates per detected person ID
        self.pos_x, self.pos_z = {}, {}

        # -----------------------------
        # Setup MediaPipe Pose model
        # -----------------------------
        # Aliases for easier access to MediaPipe classes
        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

        # Configure the pose detection model with runtime and accuracy parameters
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,  # frame-by-frame (vs. live stream)
            num_poses=max_people,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )

        # Create the pose landmarker instance using the above configuration
        self.landmarker = PoseLandmarker.create_from_options(options)

        # -----------------------------
        # Setup periodic timer callback
        # -----------------------------
        # Timer period = 1/30 seconds (≈30 FPS)
        self.timer_period = 1.0 / 30.0
        # Keep track of the frame timestamp in milliseconds for MediaPipe
        self.t_ms = 0
        self.ms_per_frame = int(1000 / 30)
        # Create timer that calls process_frame() at the specified rate
        self.timer = self.create_timer(self.timer_period, self.process_frame)

        self.get_logger().info("PoseAppNode initialized and waiting for camera frames...")

    # ============================================================
    # MAIN LOOP: process_frame()
    # Called at ~30 Hz by ROS timer
    # Grabs latest frame, detects poses, estimates 3D positions, and visualizes output
    # ============================================================
    def process_frame(self):
        """Process the most recent RGB frame and update TF/visualizations."""
        frame = self.camera.rgb_frame
        if frame is None:
            # No frame yet — camera might still be initializing
            return

        h, w = frame.shape[:2]  # Image dimensions

        # -----------------------------
        # Convert frame to MediaPipe format
        # -----------------------------
        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        # -----------------------------
        # Run pose detection
        # -----------------------------
        # detect_for_video() uses timestamps to optimize tracking
        res = self.landmarker.detect_for_video(mp_img, self.t_ms)
        self.t_ms += self.ms_per_frame  # Increment timestamp per frame

        # -----------------------------
        # Prepare visualization frame
        # -----------------------------
        annotated = frame.copy()
        # Get detected people (list of landmarks per person)
        people = res.pose_landmarks or []

        # -----------------------------
        # Process each detected person
        # -----------------------------
        for pid, lm_list in enumerate(people):
            # Draw the detected human skeleton on the output frame
            SkeletonDrawer.draw(annotated, lm_list)

            # Compute bounding box of detected person based on landmarks
            x0, y0, x1, y1 = TFCalculator.bbox(lm_list, w, h)
            # Compute bounding box center
            cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5

            # Estimate shoulder width (in pixels)
            try:
                l_sh, r_sh = lm_list[11], lm_list[12]  # Left and right shoulder landmarks
                shoulder_px = abs((l_sh.x - r_sh.x) * w)
            except Exception:
                # Fallback: use bounding box width if shoulders are not detected
                shoulder_px = max(1.0, x1 - x0)

            # Estimate depth (Z) and focal length using the shoulder width
            est = TFCalculator.estimate_distance(shoulder_px, w)
            if est is None:
                continue  # Skip if distance estimation failed

            Z, f_px = est  # Z = distance in meters, f_px = focal length in pixels

            # Convert pixel X-coordinate to world-space X (meters)
            X = TFCalculator.pixel_to_world(cx, w, Z, f_px)

            # Apply smoothing filters to stabilize positions over time
            self.pos_x[pid] = TFCalculator.smooth(self.pos_x.get(pid), X)
            self.pos_z[pid] = TFCalculator.smooth(self.pos_z.get(pid), Z)

            # -----------------------------
            # Draw visual indicators (HUD)
            # -----------------------------
            hud = f"ID {pid}  X={self.pos_x[pid]:+.2f} m  Z={self.pos_z[pid]:.2f} m"
            # Draw bounding box
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (255, 255, 0), 2)
            # Draw center point
            cv2.circle(annotated, (int(cx), int(cy)), 4, (0, 255, 255), -1)
            # Display HUD text above the bounding box
            cv2.putText(
                annotated, hud, (x0, max(20, y0 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 220, 255), 2, cv2.LINE_AA
            )

            # Update the TF plot visualization for this person
            self.tf_plot.update_person(pid, float(self.pos_x[pid]), float(self.pos_z[pid]))

        # -----------------------------
        # Draw overall info (number of people)
        # -----------------------------
        cv2.putText(
            annotated, f"People: {len(people)}", (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
        )

        # Display the annotated image
        cv2.imshow("Skeleton + TF positions", annotated)
        # Step the TF plot to refresh its display
        self.tf_plot.step()

        # Allow user to quit gracefully by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info("Quitting PoseAppNode.")
            rclpy.shutdown()

    # ============================================================
    # Cleanup: destroy_node()
    # Ensures OpenCV windows close cleanly on shutdown
    # ============================================================
    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


# ============================================================
# Entry point: main()
# Initializes ROS2, starts PoseAppNode, and spins the event loop
# ============================================================
def main(args=None):
    rclpy.init(args=args)
    node = PoseAppNode()
    try:
        rclpy.spin(node)  # Keeps node alive and responsive
    except KeyboardInterrupt:
        pass  # Handle Ctrl+C exit gracefully
    node.destroy_node()
    rclpy.shutdown()


# Run main() if executed as a script
if __name__ == '__main__':
    main()
