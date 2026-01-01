import cv2
# Wrapper around OpenCV VideoCapture for webcam input
class Webcam:
    def __init__(self, index=0, width=640, height=480, fourcc='MJPG'):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open webcam index {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def release(self):
        self.cap.release()
