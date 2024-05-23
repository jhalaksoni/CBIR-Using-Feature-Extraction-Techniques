
from django.contrib import admin
from django.urls import reverse
from django.utils.html import format_html
# Register your models here.
from app.models import ProcessedVideo


@admin.register(ProcessedVideo)
class ProcessedVideoModelAdmin(admin.ModelAdmin):
     list_display=['id','video_id','title','processed_video','thumbnail_file','created_at']
     


