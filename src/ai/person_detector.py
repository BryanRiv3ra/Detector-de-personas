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
        self.USER_ID = '2l2ybivicfey'
        self.APP_ID = 'RetoP3'
        self.WORKFLOW_ID = 'workflow-3448e7'  # Cambiamos a workflow
        self.MODEL_ID = '1580bb1932594c93b7e2e04456af7c6f'
        self.MODEL_VERSION_ID = 'latest'

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
            
            # Crear solicitud para Clarifai
            request = service_pb2.PostWorkflowResultsRequest(
                user_app_id=resources_pb2.UserAppIDSet(
                    user_id=self.USER_ID,
                    app_id=self.APP_ID
                ),
                workflow_id=self.WORKFLOW_ID,
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
            response = self.stub.PostWorkflowResults(
                request,
                metadata=self._get_metadata()
            )

            if response.status.code != status_code_pb2.SUCCESS:
                raise Exception(f"Error en la solicitud: {response.status.description}")

            # Procesar resultados
            boxes = []
            hay_persona = False
            
            # Modificación en el procesamiento de la respuesta
            for workflow_result in response.results[0].outputs:
                if workflow_result.model.id == self.MODEL_ID:
                    for region in workflow_result.data.regions:
                        for concept in region.data.concepts:
                            if concept.name == "person" and concept.value >= 0.95:
                                hay_persona = True
                                box = region.region_info.bounding_box
                                h, w = image.shape[:2]
                                boxes.append([
                                    int(box.left_col * w),
                                    int(box.top_row * h),
                                    int((box.right_col - box.left_col) * w),
                                    int((box.bottom_row - box.top_row) * h)
                                ])

            return hay_persona, np.array(boxes)

        except Exception as e:
            print(f"Error en la detección: {str(e)}")
            return False, np.array([])