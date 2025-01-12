from django.shortcuts import render
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from onboarding.views import *
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

router = DefaultRouter()
router.register(r'users', UserViewSet, basename='user')

urlpatterns = [
    path('', include(router.urls)),
    ]