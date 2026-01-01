from pathlib import Path
from include.pose_model import PoseModel
from include.pose_app_original import PoseApp
from include.webcam import Webcam
from include.tf_plot import TFPlot
from include.skeleton_drawer import SkeletonDrawer
from include.tf_calculator import TFCalculator

# ----------------------------
# Model settings
# ----------------------------
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    # "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    # "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    # "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    # "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
MODEL_PATH = Path("pose_landmarker_full.task")

# ----------------------------
# Entry point
# ----------------------------
def main():
    # Download / ensure model
    model = PoseModel(MODEL_PATH, MODEL_URL)
    
    # Create and run the pose app (camera index 0)
    app = PoseApp(cam_index=6, model_path=model.path(), MAX_PEOPLE=4)
    app.run()

if __name__ == "__main__":
    main()
