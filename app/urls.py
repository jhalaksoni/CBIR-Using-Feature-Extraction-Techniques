from django.contrib import admin
from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings


urlpatterns = [
    path('',views.CBIRView.as_view(),name='home'),

    path('rgb/', views.RgbView.as_view(), name='rgb'),
    path('gabor/', views.GaborView.as_view(), name='gabor'),
    path('daisy/', views.DaisyView.as_view(), name='daisy'),
    path('edge/', views.EdgeView.as_view(), name='edge'),
    path('lens/', views.LensView.as_view(), name='lens'),
    path('vggnet/', views.VggnetView.as_view(), name='vggnet'),
    path('resnet/', views.ResnetView.as_view(), name='resnet'),
    
    path('color/', views.ColorView.as_view(), name='color'),
    path('shape/', views.ShapeView.as_view(), name='shape'),
    path('texture/', views.TextureView.as_view(), name='texture'),
    path('deep/', views.DeepView.as_view(), name='deep'),

    # path('lens/', views.LensView.as_view({'get': 'get', 'post': 'post'}), name='lens'),
] + static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)