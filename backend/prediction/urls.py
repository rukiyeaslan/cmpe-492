from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import *

router = DefaultRouter()

urlpatterns = [
    path('predict/', PredictTransactionView.as_view(), name='predict-transaction'),
]
