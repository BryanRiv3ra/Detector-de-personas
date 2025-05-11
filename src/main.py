from camera.webcam_capture import WebcamCapture
from ai.person_detector import PersonDetector
import cv2

def main():
    # Inicializar componentes
    webcam = WebcamCapture()
    detector = PersonDetector()

    try:
        while True:
            # Capturar frame
            frame = webcam.capture_frame()
            if frame is None:
                print("Error al capturar imagen")
                break

            # Detectar personas
            hay_persona, boxes = detector.detect_person(frame)

            # Dibujar resultados
            for (x, y, w, h) in boxes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Mostrar frame
            cv2.imshow('Detector de Personas', frame)

            # Salir si se presiona 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Limpieza
        webcam.release_camera()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()