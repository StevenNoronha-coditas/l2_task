from django.urls import path
from . import views

urlpatterns = [
    path('store/', views.store_embeddings_api, name='store_embeddings'),
    path('get_response/', views.llm_call, name="get_response")
]