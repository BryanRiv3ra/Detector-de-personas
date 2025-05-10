from camera.webcam_capture import WebcamCapture
from ai.person_detector import PersonDetector
from utils.image_processor import ImageProcessor
import cv2

def main():
    # Inicializar componentes
    webcam = WebcamCapture()
    detector = PersonDetector()
    processor = ImageProcessor()

    try:
        while True:
            # Capturar frame
            frame = webcam.capture_frame()
            if frame is None:
                print("Error al capturar imagen")
                break

            # Preprocesar imagen
            processed_frame = processor.preprocess_image(frame)

            # Detectar personas
            hay_persona, boxes = detector.detect_person(processed_frame)

            # Dibujar resultados
            result_frame = processor.draw_detection_results(processed_frame, boxes)

            # Mostrar resultado
            cv2.putText(result_frame, 
                       f"Personas detectadas: {'Si' if hay_persona else 'No'}", 
                       (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       1, 
                       (0, 255, 0), 
                       2)

            # Mostrar frame
            cv2.imshow('Detector de Personas', result_frame)

            # Salir si se presiona 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Limpieza
        webcam.release_camera()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()