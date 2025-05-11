from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
import numpy as np
import cv2

class PersonDetector:
    def __init__(self):
        # Inicializar el canal y stub de Clarifai
        channel = ClarifaiChannel.get_grpc_channel()
        self.stub = service_pb2_grpc.V2Stub(channel)
        
        # Configuración de Clarifai
        self.PAT = '71f9c480b0e54848be4cc13d81242206'
        self.USER_ID = 'clarifai'  # Cambiar a clarifai
        self.APP_ID = 'main'       # Cambiar a main
        self.MODEL_ID = 'general-image-detection'
        self.MODEL_VERSION_ID = '1580bb1932594c93b7e2e04456af7c6f'

    def _get_metadata(self):
        return (('authorization', f'Key {self.PAT}'),)

    def _prepare_image(self, image):
        success, encoded_image = cv2.imencode('.jpg', image)
        if not success:
            raise Exception("Error al codificar la imagen")
        return encoded_image.tobytes()

    def detect_person(self, image):
        try:
            # Preparar imagen para Clarifai
            image_bytes = self._prepare_image(image)
            
            # Crear solicitud para Clarifai (cambiar a PostModelOutputs en lugar de PostWorkflowResults)
            request = service_pb2.PostModelOutputsRequest(
                user_app_id=resources_pb2.UserAppIDSet(
                    user_id=self.USER_ID,
                    app_id=self.APP_ID
                ),
                model_id=self.MODEL_ID,
                version_id=self.MODEL_VERSION_ID,
                inputs=[
                    resources_pb2.Input(
                        data=resources_pb2.Data(
                            image=resources_pb2.Image(
                                base64=image_bytes
                            )
                        )
                    )
                ]
            )

            # Obtener predicción
            response = self.stub.PostModelOutputs(
                request,
                metadata=self._get_metadata()
            )

            if response.status.code != status_code_pb2.SUCCESS:
                raise Exception(f"Error en la solicitud: {response.status.description}")

            # Procesar resultados
            boxes = []
            hay_persona = False
            
            # Lista ampliada de conceptos relacionados con personas
            conceptos_persona = ['person', 'people', 'adult', 'man', 'woman', 'child', 'portrait']
            
            # Procesar regiones y conceptos
            for region in response.outputs[0].data.regions:
                for concept in region.data.concepts:
                    if concept.name in conceptos_persona and concept.value >= 0.85:
                        hay_persona = True
                        box = region.region_info.bounding_box
                        h, w = image.shape[:2]
                        boxes.append([
                            int(box.left_col * w),
                            int(box.top_row * h),
                            int((box.right_col - box.left_col) * w),
                            int((box.bottom_row - box.top_row) * h)
                        ])

            # Agregar debug
            print("Conceptos detectados:", [
                (concept.name, concept.value) 
                for region in response.outputs[0].data.regions 
                for concept in region.data.concepts
            ])

            return hay_persona, np.array(boxes)

        except Exception as e:
            print(f"Error en la detección: {str(e)}")
            return False, np.array([])