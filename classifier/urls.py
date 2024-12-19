from django.urls import path
from . import views

urlpatterns = [
    path('', views.classify_email, name='classify_email'),
    path('history/', views.prediction_history, name='prediction_history'),
    path('api/classify/', views.api_classify_email, name='api_classify_email'),
]
