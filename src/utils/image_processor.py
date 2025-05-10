import cv2
import numpy as np

class ImageProcessor:
    def __init__(self):
        pass

    def preprocess_image(self, image):
        # Redimensionar la imagen si es necesario
        return cv2.resize(image, (640, 480))

    def draw_detection_results(self, image, boxes):
        # Dibujar rectángulos alrededor de las personas detectadas
        for (x, y, w, h) in boxes:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return image