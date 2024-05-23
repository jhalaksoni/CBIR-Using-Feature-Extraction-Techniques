from django.db import models
from django.core.files.uploadedfile import InMemoryUploadedFile

class ProcessedVideo(models.Model):
    video_id = models.CharField(max_length=255)
    title = models.CharField(max_length=255)
    processed_video = models.FileField(upload_to='processed_videos/')
    thumbnail_file = models.ImageField(upload_to='processed_videos/thumbnails/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

    # def save(self, *args, **kwargs):
    #     if self.thumbnail_file:
    #         # Save the thumbnail file to disk
    #         thumbnail_path = os.path.join(settings.MEDIA_ROOT, 'processed_videos', 'thumbnails')
    #         os.makedirs(thumbnail_path, exist_ok=True)
    #         thumbnail_filename = f'{self.video_id}.png'
    #         thumbnail_file_path = os.path.join(thumbnail_path, thumbnail_filename)

    #         with open(thumbnail_file_path, 'wb') as thumbnail_file:
    #             thumbnail_file.write(self.thumbnail_file.read())

    #         # Associate the saved file with the model instance
    #         self.thumbnail_file = os.path.join('processed_videos', 'thumbnails', thumbnail_filename)

    #     super().save(*args, **kwargs)