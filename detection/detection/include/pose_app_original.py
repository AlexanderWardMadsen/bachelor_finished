import mediapipe as mp
import cv2

from include.webcam import Webcam
from include.tf_plot_original import TFPlot
from include.skeleton_drawer import SkeletonDrawer
from include.tf_calculator import TFCalculator
from include.pose_model import PoseModel

class PoseApp:
    def __init__(self, cam_index=0, model_path=None, MAX_PEOPLE=4):
        self.cam = Webcam(index=cam_index)
        self.tf_plot = TFPlot()
        self.pos_x, self.pos_z = {}, {}
        self.model_path = model_path

        # MediaPipe setup
        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=MAX_PEOPLE,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        self.t_ms = 0
        self.ms_per_frame = int(1000 / self.cam.fps)

    def run(self):
        print("[INFO] Running. Press 'q' to quit.")
        while True:
            frame = self.cam.read()
            if frame is None:
                break
            h, w = frame.shape[:2]

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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
                if est is None: continue
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
                break

        self.cam.release()
        cv2.destroyAllWindows()
        plt.ioff()
        plt.close('all')
