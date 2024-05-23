from django.contrib import admin
from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings


urlpatterns = [
    path('',views.CBIRView.as_view(),name='home'),
    # path('process_video/', views.ProcessVideoView.as_view(), name='process_video'),
    # path('video_library/', views.VideoLibraryView.as_view(), name='video_library'),
    # path('realtime_video/', views.RealtimeVideoView.as_view(), name='realtime_video'),
    # path('upload_video/', views.UploadVideoView.as_view(), name='upload_video'),
    path('lens/', views.LensView.as_view(), name='lens'),
    path('sift/', views.SiftView.as_view(), name='sift'),
    path('surf/', views.SurfView.as_view(), name='surf'),    
    path('glcm/', views.GLCMView.as_view(), name='glcm'),
    path('gabor/', views.GaborView.as_view(), name='gabor'),
    # path('lens/', views.LensView.as_view({'get': 'get', 'post': 'post'}), name='lens'),
] + static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)