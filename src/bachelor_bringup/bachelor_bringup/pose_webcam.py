"""
Webcam multi-person pose (MediaPipe Tasks) with:
- on-frame skeleton rendering
- live TF-frame plot of each person's estimated (X right, Z forward) position
- Wayland-friendly settings to avoid Qt/Wayland spam

Open in VS Code and press Run ▶. Quit with 'q'.

Notes:
- Distance is estimated from shoulder width + HFOV (pinhole camera model). It's rough
  unless you calibrate HFOV and REAL_SHOULDER_WIDTH_M for your setup.
"""

import os
# --- Wayland / Qt noise workaround BEFORE importing cv2/matplotlib ---
# Use XCB for any Qt bits OpenCV might load (reduces "wayland" plugin spam).
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

# Matplotlib: prefer Tk backend (avoids Qt entirely for the plot window).
import matplotlib
matplotlib.use("TkAgg")

import math
from pathlib import Path
import urllib.request
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, deque
from mediapipe.framework.formats import landmark_pb2

# ----------------------------
# Tunables (adjust to your camera/scene)
# ----------------------------
HFOV_DEG = 60.0               # camera horizontal FOV in degrees (typical laptop cam ~60–78°)
REAL_SHOULDER_WIDTH_M = 0.38  # average adult shoulder width (m). Tweak per subject for better Z.
MAX_PEOPLE = 4                # detection cap (1..N)
SMOOTH_ALPHA = 0.35           # 0(no smooth) .. 1(strong); exponential smoothing of (X,Z)

# Model (lite = fastest)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
MODEL_PATH = Path("pose_landmarker_lite.task")


# ----------------------------
# make sure model is downloaded and if not, download it
# ----------------------------
def ensure_model(path: Path, url: str) -> str:
    if not path.exists():
        print(f"[INFO] Downloading model to {path} ...")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(path))
        print("[INFO] Download complete.")
    return str(path)

# ----------------------------
# convert BRG frame cv2 -> mediapipe Image rgb
# ----------------------------
def mpimage_from_bgr(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

# ----------------------------
# convert list[NormalizedLandmark] -> NormalizedLandmarkList mediapipe proto (needed for draw_landmarks)
# ----------------------------
def to_landmark_list(lm_list):
    """Convert list[NormalizedLandmark] -> NormalizedLandmarkList proto (needed for draw_landmarks)."""
    lml = landmark_pb2.NormalizedLandmarkList()
    for lm in lm_list:
        p = lml.landmark.add()
        p.x, p.y, p.z = lm.x, lm.y, lm.z
        if hasattr(lm, "visibility"):
            p.visibility = lm.visibility
    return lml

# ----------------------------
# draw skeleton on image
# ----------------------------
def draw_skeleton(image_bgr, lm_list):
    """Draw 33-keypoint skeleton for a single person."""
    du = mp.solutions.drawing_utils
    du.draw_landmarks(
        image=image_bgr,
        landmark_list=to_landmark_list(lm_list),
        connections=mp.solutions.pose.POSE_CONNECTIONS,
        landmark_drawing_spec=du.DrawingSpec(thickness=2, circle_radius=2),
        connection_drawing_spec=du.DrawingSpec(thickness=2),
    )

# ----------------------------
# compute bbox from landmarks (bounding box)
# ----------------------------
def bbox_from_landmarks(lms, w, h):
    xs = [max(0.0, min(1.0, lm.x)) * w for lm in lms]
    ys = [max(0.0, min(1.0, lm.y)) * h for lm in lms]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

# ----------------------------
# estimate distance (Z) in meters from pixel width of shoulders
# ----------------------------
def estimate_distance_m(pixel_width, frame_width, hfov_deg, real_width_m=REAL_SHOULDER_WIDTH_M):
    """
    Pinhole model:
      f(px) = (W/2) / tan(HFOV/2),  Z = (real_width * f) / pixel_width
    """
    if pixel_width <= 1e-6:
        return None
    # focal length in pixels
    f_px = (frame_width * 0.5) / math.tan(math.radians(hfov_deg * 0.5))
    z_m = (real_width_m * f_px) / pixel_width
    return z_m, f_px

# ----------------------------
# exponential smoothing to reduce jitter in (X,Z) estimates
# ----------------------------
def smooth(prev, new, a=SMOOTH_ALPHA):
    if prev is None: return new
    return (1 - a) * prev + a * new

# ----------------------------
# Live TF plot (top-down X vs Z)
# ----------------------------
class TFPlot:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Estimated TF Positions (top-down)")
        self.ax.set_xlabel("X (right, meters)")
        self.ax.set_ylabel("Z (forward, meters)")
        self.ax.grid(True)
        self.scatters = {}   # pid -> PathCollection
        self.texts = {}      # pid -> Text
        self.history = defaultdict(lambda: deque(maxlen=20))  # small trail per ID

        # initial limits; will auto-extend as needed
        self.xmin, self.xmax = -2, 2
        self.zmin, self.zmax = 0, 6
        self.ax.set_xlim(self.xmin, self.xmax)
        self.ax.set_ylim(self.zmin, self.zmax)

    # ----------------------------
    # auto-extend plot limits if needed
    # ----------------------------
    def _maybe_expand_limits(self, x, z):
        changed = False
        margin = 0.5
        if x < self.xmin + margin: self.xmin = x - margin; changed = True
        if x > self.xmax - margin: self.xmax = x + margin; changed = True
        if z < self.zmin + margin: self.zmin = max(0, z - margin); changed = True
        if z > self.zmax - margin: self.zmax = z + margin; changed = True
        if changed:
            self.ax.set_xlim(self.xmin, self.xmax)
            self.ax.set_ylim(self.zmin, self.zmax)

    # ----------------------------
    # update person position by ID and draw line between last positions
    # ----------------------------
    def update_person(self, pid, x, z):
        self._maybe_expand_limits(x, z)
        self.history[pid].append((x, z))

        trail = np.array(self.history[pid])
        # draw/update scatter
        if pid not in self.scatters:
            self.scatters[pid] = self.ax.scatter([x], [z], s=40)
        else:
            offsets = self.scatters[pid].get_offsets()
            offsets[:] = np.array([[x, z]])
            self.scatters[pid].set_offsets(offsets)

        # draw/update label
        label = f"ID {pid}\nX={x:+.2f} m\nZ={z:.2f} m"
        if pid not in self.texts:
            self.texts[pid] = self.ax.text(x, z, label, fontsize=8, va="bottom", ha="left")
        else:
            self.texts[pid].set_position((x, z))
            self.texts[pid].set_text(label)

        # draw short trail
        if len(trail) > 1:
            self.ax.plot(trail[:,0], trail[:,1], linewidth=1, alpha=0.4)

    # ----------------------------
    # refresh the matplotlib window
    # ----------------------------
    def step(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

# ----------------------------
# Main
# ----------------------------
def main():
    model_path = ensure_model(MODEL_PATH, MODEL_URL)

    # Webcam (plain index on Linux)
    cap = cv2.VideoCapture(6)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    # ----------------------------
    # initial camera setup
    # ----------------------------
    # keep decode cost modest
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    # ----------------------------
    # MediaPipe Tasks setup
    # ----------------------------
    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

    # ----------------------------
    # Create a PoseLandmarker instance with the model, and confidence thresholds and disables segmentation masks for speed.
    # ----------------------------
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,     # timestamps enable tracking
        num_poses=MAX_PEOPLE,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    # plot
    tf_plot = TFPlot()

    # state: smoothed positions by (frame-local) ID index
    pos_x = {}
    pos_z = {}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps < 1: fps = 30.0
    ms_per_frame = int(1000 / fps)
    t_ms = 0

    print("[INFO] Running. Press 'q' to quit.")
    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]

            # inference
            mp_img = mpimage_from_bgr(frame)
            res = landmarker.detect_for_video(mp_img, t_ms)
            t_ms += ms_per_frame

            annotated = frame.copy()

            # draw skeletons + compute TF positions
            people = res.pose_landmarks or []
            for pid, lm_list in enumerate(people):
                # draw skeleton (fix: convert list -> proto)
                draw_skeleton(annotated, lm_list)

                # bbox & center (pixels)
                x0, y0, x1, y1 = bbox_from_landmarks(lm_list, w, h)
                cx = (x0 + x1) * 0.5
                cy = (y0 + y1) * 0.5

                # shoulder pixel width
                try:
                    l_sh = lm_list[11]; r_sh = lm_list[12]
                    shoulder_px = abs((l_sh.x - r_sh.x) * w)
                except Exception:
                    shoulder_px = max(1.0, x1 - x0)

                # distance (Z) + focal length
                est = estimate_distance_m(shoulder_px, w, HFOV_DEG, REAL_SHOULDER_WIDTH_M)
                if est is None:
                    continue
                Z, f_px = est

                # horizontal displacement (pixels, right positive)
                x_px = cx - (w * 0.5)
                # pinhole: X = (x_px * Z) / f
                X = (x_px * Z) / max(f_px, 1e-6)

                # smooth
                pos_x[pid] = smooth(pos_x.get(pid), X)
                pos_z[pid] = smooth(pos_z.get(pid), Z)

                # HUD on video
                bearing_deg = ( (cx / w) * 2.0 - 1.0 ) * (HFOV_DEG / 2.0)
                hud = f"ID {pid}  X={pos_x[pid]:+.2f} m  Z={pos_z[pid]:.2f} m  bearing={bearing_deg:+.1f}°"
                cv2.rectangle(annotated, (x0, y0), (x1, y1), (255, 255, 0), 2)
                cv2.circle(annotated, (int(cx), int(cy)), 4, (0, 255, 255), -1)
                cv2.putText(annotated, hud, (x0, max(20, y0 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 220, 255), 2, cv2.LINE_AA)

                # update plot
                tf_plot.update_person(pid, float(pos_x[pid]), float(pos_z[pid]))

            # show windows
            cv2.putText(annotated, f"People: {len(people)}", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Skeleton + TF positions (Webcam)", annotated)

            tf_plot.step()  # refresh the matplotlib window

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.close('all')


if __name__ == "__main__":
    main()
