import cv2
import mediapipe as mp

class PoseDetector:
    """Runs MediaPipe pose detection and returns the nose pixel position."""

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5)
        self.conf_threshold = 0.5

    def detect_nose(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        if not results.pose_landmarks:
            return None, 0.0
        nose = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        if nose.visibility < self.conf_threshold:
            return None, nose.visibility

        h, w, _ = frame.shape
        cx, cy = int(nose.x * w), int(nose.y * h)
        return (cx, cy), nose.visibility
