from .serializers import *
from .models import *
from drf_spectacular.utils import extend_schema
from rest_framework.views import APIView
from django.conf import settings
from drf_spectacular.utils import extend_schema
from django.http import JsonResponse
from frauddetection.randomForest import FraudDetectionModel

# Load the model once during server startup
fraud_model = FraudDetectionModel.load_model(settings.MODEL_PATH)

class PredictTransactionView(APIView):
    serializer_class = TransactionPredictionSerializer
    basename = 'predict' 

    @extend_schema(
        description="Predict if a transaction is fraudulent",
        request=TransactionPredictionSerializer,
        responses={
            200: {
                'description': 'Prediction successful',
                'content': {
                    'application/json': {
                        'schema': {
                            'type': 'object',
                            'properties': {
                                'is_fraud': {'type': 'boolean'},
                                'probability': {'type': 'number'}
                            }
                        }
                    }
                }
            },
            400: {'description': 'Invalid request data'},
            500: {'description': 'Server error'}
        },
        tags=['Fraud Detection']
    )
    def post(self, request):
        try:
            serializer = self.serializer_class(data=request.data)
            if not serializer.is_valid():
                return JsonResponse({"error": serializer.errors}, status=400)

            # Use the loaded model to predict
            is_fraud, probability = fraud_model.predict(serializer.validated_data)

            # Return the prediction as JSON
            return JsonResponse({
                "is_fraud": is_fraud,
                "probability": probability
            })
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)