"""
Webcam multi-person pose (OpenPose Full-Body) with:
- on-frame skeleton rendering
- live top-down plot of each person's estimated (X right, Z forward) position
- focus on legs and feet (Full-Body model)
- low-angle camera friendly
- switch between light (BODY_25) and heavy (COCO + foot) models
"""

import os
import sys
import math
import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # avoid Qt entirely
import matplotlib.pyplot as plt
from collections import defaultdict, deque

# ----------------------------
# Tunables
# ----------------------------
HFOV_DEG = 60.0               # camera horizontal FOV in degrees
REAL_SHOULDER_WIDTH_M = 0.38  # average adult shoulder width in meters
MAX_PEOPLE = 4
SMOOTH_ALPHA = 0.35
CAMERA_INDEX = 0              # webcam index

# ----------------------------
# OpenPose Python setup
# ----------------------------
# Download OpenPose binaries (https://github.com/CMU-Perceptual-Computing-Lab/openpose)
# and set OPENPOSE_ROOT to the folder containing python/openpose/...
OPENPOSE_ROOT = os.getenv("OPENPOSE_ROOT", "/path/to/openpose")
sys.path.append(os.path.join(OPENPOSE_ROOT, "build/python"))
try:
    from openpose import pyopenpose as op
except ImportError as e:
    raise RuntimeError("Error: OpenPose library not found. Set OPENPOSE_ROOT.") from e

# ----------------------------
# Model selection: 'BODY_25' = light, 'BODY_135' = full
MODEL_TYPE = "BODY_25"  # switch to 'BODY_135' for heavy full-body
params = dict()
params["model_folder"] = os.path.join(OPENPOSE_ROOT, "models")
params["model_pose"] = MODEL_TYPE
params["net_resolution"] = "-1x368" if MODEL_TYPE=="BODY_25" else "-1x368"
params["disable_multi_thread"] = True  # simpler for CPU debugging

# ----------------------------
# Distance estimator
# ----------------------------
def estimate_distance_m(pixel_width, frame_width, hfov_deg, real_width_m=REAL_SHOULDER_WIDTH_M):
    if pixel_width <= 1e-6:
        return None
    f_px = (frame_width * 0.5) / math.tan(math.radians(hfov_deg * 0.5))
    z_m = (real_width_m * f_px) / pixel_width
    return z_m, f_px

def smooth(prev, new, a=SMOOTH_ALPHA):
    if prev is None: return new
    return (1 - a) * prev + a * new

# ----------------------------
# Live top-down plot
# ----------------------------
class TFPlot:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Estimated TF Positions")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Z (m)")
        self.ax.grid(True)
        self.scatters = {}
        self.texts = {}
        self.history = defaultdict(lambda: deque(maxlen=20))
        self.xmin, self.xmax = -2, 2
        self.zmin, self.zmax = 0, 6
        self.ax.set_xlim(self.xmin, self.xmax)
        self.ax.set_ylim(self.zmin, self.zmax)

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

    def update_person(self, pid, x, z):
        self._maybe_expand_limits(x, z)
        self.history[pid].append((x, z))
        trail = np.array(self.history[pid])
        if pid not in self.scatters:
            self.scatters[pid] = self.ax.scatter([x], [z], s=40)
        else:
            self.scatters[pid].set_offsets([[x, z]])
        label = f"ID {pid}\nX={x:+.2f} m\nZ={z:.2f} m"
        if pid not in self.texts:
            self.texts[pid] = self.ax.text(x, z, label, fontsize=8, va="bottom", ha="left")
        else:
            self.texts[pid].set_position((x, z))
            self.texts[pid].set_text(label)
        if len(trail) > 1:
            self.ax.plot(trail[:,0], trail[:,1], linewidth=1, alpha=0.4)

    def step(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

# ----------------------------
# Main
# ----------------------------
def main():
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    tf_plot = TFPlot()
    pos_x, pos_z = {}, {}

    print("[INFO] Running. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok: break
        h, w = frame.shape[:2]

        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop([datum])

        annotated = datum.cvOutputData.copy()
        keypoints = datum.poseKeypoints  # shape: (people, n_joints, 3)

        if keypoints is not None:
            for pid, person in enumerate(keypoints):
                # get shoulder width
                try:
                    l_sh = person[5]; r_sh = person[2]  # BODY_25 indices: 5=L,2=R
                    shoulder_px = abs(l_sh[0] - r_sh[0])
                except Exception:
                    shoulder_px = max(1.0, np.ptp(person[:,0]))

                est = estimate_distance_m(shoulder_px, w, HFOV_DEG, REAL_SHOULDER_WIDTH_M)
                if est is None: continue
                Z, f_px = est

                cx = np.mean(person[:,0])
                x_px = cx - (w * 0.5)
                X = (x_px * Z) / max(f_px, 1e-6)

                pos_x[pid] = smooth(pos_x.get(pid), X)
                pos_z[pid] = smooth(pos_z.get(pid), Z)

                # HUD
                hud = f"ID {pid}  X={pos_x[pid]:+.2f} m  Z={pos_z[pid]:.2f} m"
                cv2.putText(annotated, hud, (10, 30 + pid*25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)

                # update plot
                tf_plot.update_person(pid, float(pos_x[pid]), float(pos_z[pid]))

        cv2.imshow("OpenPose FullBody", annotated)
        tf_plot.step()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.close('all')

if __name__ == "__main__":
    main()
