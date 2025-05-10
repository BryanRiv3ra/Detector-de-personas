import cv2

class WebcamCapture:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise Exception("No se pudo acceder a la cámara web")

    def capture_frame(self):
        ret, frame = self.camera.read()
        if not ret:
            return None
        return frame

    def release_camera(self):
        self.camera.release()