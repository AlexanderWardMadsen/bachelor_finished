import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

class SkeletonDrawer:
    # ----------------------------------
    # Draw skeleton on image using MediaPipe drawing utilities
    # ----------------------------------
    @staticmethod
    def to_landmark_list(lm_list):
        lml = landmark_pb2.NormalizedLandmarkList()
        for lm in lm_list:
            p = lml.landmark.add()
            p.x, p.y, p.z = lm.x, lm.y, lm.z
            if hasattr(lm, "visibility"):
                p.visibility = lm.visibility
        return lml

    # ----------------------------------
    # Draw skeleton on image using MediaPipe drawing utilities
    # ----------------------------------
    @staticmethod
    def draw(image_bgr, lm_list):
        du = mp.solutions.drawing_utils
        du.draw_landmarks(
            image=image_bgr,
            landmark_list=SkeletonDrawer.to_landmark_list(lm_list),
            connections=mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=du.DrawingSpec(thickness=2, circle_radius=2),
            connection_drawing_spec=du.DrawingSpec(thickness=2),
        )
