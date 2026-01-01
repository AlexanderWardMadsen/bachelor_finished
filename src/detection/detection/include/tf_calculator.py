import math
from detection.include.calculate_z import calculate_z

SMOOTH_ALPHA = 0.35
REAL_SHOULDER_WIDTH_M = 0.32 #0.38 # average human shoulder width in meters
HFOV_DEG = 87.0 #60.0
VFOV_DEG = 58.0



class TFCalculator:
    
    # ----------------------------------
    # Simple exponential smoothing
    # ----------------------------------
    @staticmethod
    def smooth(prev, new, a=SMOOTH_ALPHA):
        if prev is None: return new
        return (1 - a) * prev + a * new
    
    # ----------------------------------
    # Static methods for bounding box, distance estimation, and pixel-to-world conversion
    # ----------------------------------
    @staticmethod
    def bbox(lms, w, h):
        xs = [max(0.0, min(1.0, lm.x)) * w for lm in lms]
        ys = [max(0.0, min(1.0, lm.y)) * h for lm in lms]
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

    # ----------------------------------
    # Estimate distance using pinhole camera model
    # ----------------------------------
    @staticmethod
    def estimate_distance(pixel_width, frame_width, pixel_height, frame_height, cy, hfov_deg=HFOV_DEG, real_width=REAL_SHOULDER_WIDTH_M):
        if pixel_width <= 1e-6:
            return None
        f_py = (frame_height * 0.5) / math.tan(math.radians(VFOV_DEG * 0.5))
        f_px = (frame_width * 0.5) / math.tan(math.radians(hfov_deg * 0.5))
        z_m = (real_width * f_px) / pixel_width
        y = ((cy - (frame_height * 0.5)) * z_m) / max(f_py, 1e-6)
        z = calculate_z.calculate_true_z(y, z_m)
        return z, f_px

    # ----------------------------------
    # Convert pixel coordinate to world coordinate
    # ----------------------------------
    @staticmethod
    def pixel_to_world(cx, w, Z, f_px):
        x_px = cx - (w * 0.5)
        return (x_px * Z) / max(f_px, 1e-6)

    # ----------------------------------
    # make a comparitor to compare with depth camera and calculated distance
    # ----------------------------------
