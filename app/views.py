import os
import math
import pickle
import logging
import random
import string
import tempfile
import base64
import numpy as np
import cv2
import pafy
import torch
import imageio.v2 as imageio
from scipy.ndimage import convolve
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from django.shortcuts import render, redirect, HttpResponseRedirect
from django.views import View
from django.urls import reverse
from django.conf import settings
from django.core.files import File
from django.core.files.storage import default_storage
from django.core.files.temp import NamedTemporaryFile
from django.contrib import messages
from django.contrib.messages import get_messages
from django.http import JsonResponse, FileResponse, StreamingHttpResponse, HttpResponseBadRequest, HttpResponseServerError
from app.models import ProcessedVideo
from six.moves import cPickle
from torchvision import models
from torchvision.transforms import transforms
from torch.autograd import Variable

from skimage import io, color, feature
from skimage.filters import gabor_kernel
from skimage.feature import graycomatrix, graycoprops, match_descriptors
from .src.evaluate import evaluate_class
from .src.DB import Database
from PIL import Image

from torchvision import transforms
from .src.daisy import Daisy
from .src.HOG import HOG
from .src.vggnet import VGGNet
from .src.resnet import ResidualNet
from .src.gabor import Gabor
from .src.edge import Edge
from .src.color import Color

logger = logging.getLogger(__name__)

class GaborViews(View):
    http_method_names = ['get', 'post']

    def get(self, request):
        return render(request, 'app/gabor.html')

    def post(self, request):
        # Get the uploaded query image file
        query_image_file = request.FILES.get('query_image')
        if query_image_file:
            # Create the 'temp' directory if it doesn't exist
            temp_dir = os.path.join(os.getcwd(), 'temp')
            os.makedirs(temp_dir, exist_ok=True)

            # Save the uploaded file to the temporary location
            query_image_path = os.path.join(temp_dir, query_image_file.name)
            with open(query_image_path, 'wb+') as f:
                for chunk in query_image_file.chunks():
                    f.write(chunk)

            # Load the dataset features from the PKL file
            pkl_file = os.path.join(settings.BASE_DIR, 'app/models/gabor.pkl')
            with open(pkl_file, 'rb') as f:
                features, labels = pickle.load(f)

            # Extract features from the query image using Gabor
            query_image_array = io.imread(query_image_path)
            gabor = Gabor()
            query_features = gabor.gabor_histogram(query_image_array, type=gabor.h_type, n_slice=gabor.n_slice)

            # Calculate similarity between the query features and pre-computed features
            similarities = []
            for feature, label in zip(features, labels):
                similarity = np.dot(query_features, feature) / (np.linalg.norm(query_features) * np.linalg.norm(feature))
                similarities.append((label, similarity))

            # Sort images based on similarity
            sorted_images = sorted(similarities, key=lambda x: x[1], reverse=True)

            # Render the template with the sorted images
            context = {
                'sorted_images': sorted_images,
                'query_image_url': os.path.join(settings.STATIC_URL, 'app/database', query_image_file.name),
            }
            return render(request, 'app/gabor.html', context)
        else:
            # Handle the case when no image file is uploaded
            return render(request, 'app/gabor.html', {'error_message': 'Please upload an image file.'})
        





class VggnetViews(View):
    http_method_names = ['get', 'post']

    def get(self, request):
        return render(request, 'app/vggnet.html')

    def post(self, request):
        # Get the uploaded query image file
        query_image_file = request.FILES.get('query_image')
        if query_image_file:
            # Create the 'temp' directory if it doesn't exist
            temp_dir = os.path.join(os.getcwd(), 'temp')
            os.makedirs(temp_dir, exist_ok=True)

            # Save the uploaded file to the temporary location
            query_image_path = os.path.join(temp_dir, query_image_file.name)
            with open(query_image_path, 'wb+') as f:
                for chunk in query_image_file.chunks():
                    f.write(chunk)

            # Load the dataset features from the PKL file
            pkl_file = os.path.join(settings.BASE_DIR, 'app/models/vgg.pkl')
            
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                features = data['features']
                labels = data['labels']

            # Load the VGG model
            model = VGGNet(weights=models.VGG19_Weights.IMAGENET1K_V1, requires_grad=False, model='vgg19')
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()

            # Extract features from the query image
            query_image_array = io.imread(query_image_path)
            query_image_array = np.transpose(query_image_array, (2, 0, 1)) / 255.
            query_image_array[0] -= 103.939 / 255.
            query_image_array[1] -= 116.779 / 255.
            query_image_array[2] -= 123.68 / 255.
            query_image_array = np.expand_dims(query_image_array, axis=0)
            query_image_tensor = torch.from_numpy(query_image_array).float()
            if torch.cuda.is_available():
                query_image_tensor = query_image_tensor.cuda()

            with torch.no_grad():
                query_features = model(query_image_tensor)['avg'].cpu().numpy()
            query_features = query_features.flatten()

            # Calculate similarity between the query features and pre-computed features
            similarities = []
            for feature, label in zip(features, labels):
                similarity = np.dot(query_features, feature) / (np.linalg.norm(query_features) * np.linalg.norm(feature))
                similarities.append((label, similarity))

            # Sort images based on similarity
            sorted_images = sorted(similarities, key=lambda x: x[1], reverse=True)

            # Render the template with the sorted images
            context = {
                'sorted_images': sorted_images,
                'query_image_url': os.path.join(settings.STATIC_URL, 'app/database', query_image_file.name),
            }
            return render(request, 'app/vggnet.html', context)
        else:
            # Handle the case when no image file is uploaded
            return render(request, 'app/vggnet.html', {'error_message': 'Please upload an image file.'})




class ResnetViews(View):
    http_method_names = ['get', 'post']

    def get(self, request):
        return render(request, 'app/resnet.html')

    def post(self, request):
        # Get the uploaded query image file
        query_image_file = request.FILES.get('query_image')
        if query_image_file:
            # Create the 'temp' directory if it doesn't exist
            temp_dir = os.path.join(os.getcwd(), 'temp')
            os.makedirs(temp_dir, exist_ok=True)

            # Save the uploaded file to the temporary location
            query_image_path = os.path.join(temp_dir, query_image_file.name)
            with open(query_image_path, 'wb+') as f:
                for chunk in query_image_file.chunks():
                    f.write(chunk)

            # Load the dataset features from the PKL file
            pkl_file = os.path.join(settings.BASE_DIR, 'app/models/resnet.pkl')
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                features = data['features']
                labels = data['labels']

            # Load the ResidualNet model
            model = ResidualNet(model='resnet152', pretrained=True)
            model.eval()

            # Define preprocessing transformations
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            # Read the query image, convert to PIL image, and apply preprocessing
            query_image = imageio.imread(query_image_path)
            query_image = Image.fromarray(query_image)
            query_image = preprocess(query_image)
            query_image = query_image.unsqueeze(0)  # add batch dimension

            if torch.cuda.is_available():
                model = model.cuda()
                query_image = query_image.cuda()

            with torch.no_grad():
                query_features = model(query_image)['avg']
                query_features = query_features.cpu().numpy().flatten()

            # Normalize query features
            query_features = query_features / np.linalg.norm(query_features)

            # Calculate similarity between the query features and pre-computed features
            similarities = []
            for feature, label in zip(features, labels):
                feature = feature / np.linalg.norm(feature)  # Normalize database features
                similarity = np.dot(query_features, feature)
                similarities.append((label, similarity))

            # Sort images based on similarity
            sorted_images = sorted(similarities, key=lambda x: x[1], reverse=True)

            # Render the template with the sorted images
            context = {
                'sorted_images': sorted_images,
                'query_image_url': os.path.join(settings.STATIC_URL, 'app/database', query_image_file.name),
            }
            return render(request, 'app/resnet.html', context)
        else:
            # Handle the case when no image file is uploaded
            return render(request, 'app/resnet.html', {'error_message': 'Please upload an image file.'})

class RgbView(View):
    http_method_names = ['get', 'post']

    def get(self, request):
        return render(request, 'app/rgb.html')

    def post(self, request):
        # Get the uploaded query image file
        query_image_file = request.FILES.get('query_image')
        if query_image_file:
            # Create the 'temp' directory if it doesn't exist
            temp_dir = os.path.join(os.getcwd(), 'temp')
            os.makedirs(temp_dir, exist_ok=True)

            # Save the uploaded file to the temporary location
            query_image_path = os.path.join(temp_dir, query_image_file.name)
            with open(query_image_path, 'wb+') as f:
                for chunk in query_image_file.chunks():
                    f.write(chunk)

            # Load the dataset features from the PKL file
            pkl_file = os.path.join(settings.BASE_DIR, 'app/models/color.pkl')
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                features = data['features']
                labels = data['labels']

            # Extract features from the query image
            query_image_array = io.imread(query_image_path)
            color_extractor = Color()
            query_features = color_extractor.histogram(query_image_array, type=color_extractor.h_type, n_slice=color_extractor.n_slice)

            # Calculate similarity between the query features and pre-computed features
            similarities = []
            for feature, label in zip(features, labels):
                similarity = np.dot(query_features, feature) / (np.linalg.norm(query_features) * np.linalg.norm(feature))
                similarities.append((label, similarity))

            # Sort images based on similarity
            sorted_images = sorted(similarities, key=lambda x: x[1], reverse=True)

            # Render the template with the sorted images
            context = {
                'sorted_images': sorted_images,
                'query_image_url': os.path.join(settings.STATIC_URL, 'app/database', query_image_file.name),
            }
            return render(request, 'app/rgb.html', context)
        else:
            # Handle the case when no image file is uploaded
            return render(request, 'app/rgb.html', {'error_message': 'Please upload an image file.'})

class EdgeViews(View):
    http_method_names = ['get', 'post']

    def get(self, request):
        return render(request, 'app/edge.html')

    def post(self, request):
        # Get the uploaded query image file
        query_image_file = request.FILES.get('query_image')
        if query_image_file:
            # Create the 'temp' directory if it doesn't exist
            temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
            os.makedirs(temp_dir, exist_ok=True)

            # Save the uploaded file to the temporary location
            query_image_path = os.path.join(temp_dir, query_image_file.name)
            with open(query_image_path, 'wb+') as f:
                for chunk in query_image_file.chunks():
                    f.write(chunk)

            # Load the dataset features from the PKL file
            pkl_file = os.path.join(settings.BASE_DIR, 'app/models/edge.pkl')
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                features = data['features']
                labels = data['labels']

            # Extract features from the query image
            query_image_array = io.imread(query_image_path)
            edge = Edge()
            query_features = edge.histogram(query_image_array, stride=edge.stride, type=edge.h_type, n_slice=edge.n_slice)

            # Calculate similarity between the query features and pre-computed features
            similarities = []
            for feature, label in zip(features, labels):
                similarity = np.dot(query_features, feature) / (np.linalg.norm(query_features) * np.linalg.norm(feature))
                similarities.append((label, similarity))

            # Sort images based on similarity
            sorted_images = sorted(similarities, key=lambda x: x[1], reverse=True)

            # Render the template with the sorted images
            context = {
                'sorted_images': sorted_images,
                'query_image_url': os.path.join(settings.MEDIA_URL, 'temp', query_image_file.name),
            }
            return render(request, 'app/edge.html', context)
        else:
            # Handle the case when no image file is uploaded
            return render(request, 'app/edge.html', {'error_message': 'Please upload an image file.'})

class LensView(View):
    http_method_names = ['get', 'post']

    def get(self, request):
        return render(request, 'app/lens.html')

    def post(self, request):
        # Get the uploaded query image file
        query_image_file = request.FILES.get('query_image')
        if query_image_file:
            # Create the 'temp' directory if it doesn't exist
            temp_dir = os.path.join(os.getcwd(), 'temp')
            os.makedirs(temp_dir, exist_ok=True)

            # Save the uploaded file to the temporary location
            query_image_path = os.path.join(temp_dir, query_image_file.name)
            with open(query_image_path, 'wb+') as f:
                for chunk in query_image_file.chunks():
                    f.write(chunk)

            # Load the dataset features from the PKL file
            pkl_file = os.path.join(settings.BASE_DIR, 'app/models/hog.pkl')
            with open(pkl_file, 'rb') as f:
                features, labels = pickle.load(f)

            # Extract features from the query image
            query_image_array = io.imread(query_image_path)
            hog = HOG()
            query_features = hog.histogram(query_image_array, type=hog.h_type, n_slice=hog.n_slice)

            # Calculate similarity between the query features and pre-computed features
            similarities = []
            for feature, label in zip(features, labels):
                similarity = np.dot(query_features, feature) / (np.linalg.norm(query_features) * np.linalg.norm(feature))
                similarities.append((label, similarity))

            # Sort images based on similarity
            sorted_images = sorted(similarities, key=lambda x: x[1], reverse=True)

            # Render the template with the sorted images
            context = {
                'sorted_images': sorted_images,
                'query_image_url': os.path.join(settings.STATIC_URL, 'app/database', query_image_file.name),
            }
            return render(request, 'app/lens.html', context)
        else:
            # Handle the case when no image file is uploaded
            return render(request, 'app/lens.html', {'error_message': 'Please upload an image file.'})



class DaisyView(View):
    http_method_names = ['get', 'post']

    def get(self, request):
        return render(request, 'app/daisy.html')

    def post(self, request):
        # Get the uploaded query image file
        query_image_file = request.FILES.get('query_image')
        if query_image_file:
            # Create the 'temp' directory if it doesn't exist
            temp_dir = os.path.join(os.getcwd(), 'temp')
            os.makedirs(temp_dir, exist_ok=True)

            # Save the uploaded file to the temporary location
            query_image_path = os.path.join(temp_dir, query_image_file.name)
            with open(query_image_path, 'wb+') as f:
                for chunk in query_image_file.chunks():
                    f.write(chunk)

            # Load the dataset features from the PKL file
            pkl_file = os.path.join(settings.BASE_DIR, 'app/models/daisy.pkl')
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                features = data['features']
                labels = data['labels']

            # Extract features from the query image
            query_image_array = io.imread(query_image_path)
            daisy_extractor = Daisy()
            query_features = daisy_extractor.histogram(query_image_array, type=daisy_extractor.h_type, n_slice=daisy_extractor.n_slice)

            # Calculate similarity between the query features and pre-computed features
            similarities = []
            for feature, label in zip(features, labels):
                similarity = np.dot(query_features, feature) / (np.linalg.norm(query_features) * np.linalg.norm(feature))
                similarities.append((label, similarity))

            # Sort images based on similarity
            sorted_images = sorted(similarities, key=lambda x: x[1], reverse=True)

            # Render the template with the sorted images
            context = {
                'sorted_images': sorted_images,
                'query_image_url': os.path.join(settings.STATIC_URL, 'app/database', query_image_file.name),
            }
            return render(request, 'app/daisy.html', context)
        else:
            # Handle the case when no image file is uploaded
            return render(request, 'app/daisy.html', {'error_message': 'Please upload an image file.'})
        
class ShapeView(View):
  def get(self,request):
      
      return render(
        request,
        'app/shape.html',
        {
            
        }
      )
class TextureView(View):
  def get(self,request):
      
      return render(
        request,
        'app/texture.html',
        {
            
        }
      )
      
class ColorView(View):
  def get(self,request):
      
      return render(
        request,
        'app/color.html',
        {
            
        }
      )
      
class DeepView(View):
  def get(self,request):
      return render(
        request,
        'app/deep.html',
        {
            
        }
      )    
# Create your views here.
class CBIRView(View):
  def get(self,request):
      
      return render(
        request,
        'app/home.html',
        {
            
        }
      )
























class GaborView(View):
    http_method_names = ['get', 'post']

    @staticmethod
    def extract_gabor_features(image):
        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
            image_gray = image.squeeze()
        else:
            image_gray = color.rgb2gray(image)

        features = feature.hog(image_gray, orientations=8, pixels_per_cell=(16, 16),
                                cells_per_block=(1, 1), block_norm='L2-Hys',
                                visualize=False, feature_vector=True)
        return features.flatten()

    @staticmethod
    def compare_gabor_features(query_image_path, dataset_directory, pkl_file):
        # Load query image features
        query_image = io.imread(query_image_path)
        query_features = GaborView.extract_gabor_features(query_image)

        # Load dataset features from the PKL file
        with open(pkl_file, 'rb') as f:
            features_dict = pickle.load(f)

        similarity_scores = []

        # Loop through all images in the dataset
        for filename, dataset_features in features_dict.items():
            # Ensure both feature vectors have the same length
            min_length = min(len(query_features), len(dataset_features))
            query_features = query_features[:min_length]
            dataset_features = dataset_features[:min_length]

            # Calculate cosine similarity between query and dataset images
            similarity = cosine_similarity([query_features], [dataset_features])[0][0]
            similarity_scores.append((filename, similarity))

        # Sort images based on similarity (higher similarity first)
        sorted_images = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        return sorted_images[:10]

    def get(self, request):
        return render(request, 'app/gabor.html')

    def post(self, request):
        # Get the uploaded query image file
        query_image_file = request.FILES.get('query_image')

        if query_image_file:
            # Create the 'temp' directory if it doesn't exist
            temp_dir = os.path.join(os.getcwd(), 'temp')
            os.makedirs(temp_dir, exist_ok=True)

            # Save the uploaded file to the temporary location
            query_image_path = os.path.join(temp_dir, query_image_file.name)
            with open(query_image_path, 'wb+') as f:
                for chunk in query_image_file.chunks():
                    f.write(chunk)

            # Load the dataset features from the PKL file
            pkl_file = 'app/models/gabor.pkl'
            dataset_directory = 'media/database'

            # Call the compare_hog_features function and get the sorted images
            sorted_images = GaborView.compare_gabor_features(query_image_path, dataset_directory, pkl_file)

            # Render the template with the sorted images
            return render(request, 'app/gabor.html', {'sorted_images': sorted_images})
        else:
            # Handle the case when no image file is uploaded
            return render(request, 'app/gabor.html', {'error_message': 'Please upload an image file.'})

class EdgeView(View):
    http_method_names = ['get', 'post']

    @staticmethod
    def extract_edge_features(image):
        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
            image_gray = image.squeeze()
        else:
            image_gray = color.rgb2gray(image)

        features = feature.hog(image_gray, orientations=8, pixels_per_cell=(16, 16),
                                cells_per_block=(1, 1), block_norm='L2-Hys',
                                visualize=False, feature_vector=True)
        return features.flatten()

    @staticmethod
    def compare_edge_features(query_image_path, dataset_directory, pkl_file):
        # Load query image features
        query_image = io.imread(query_image_path)
        query_features = EdgeView.extract_edge_features(query_image)

        # Load dataset features from the PKL file
        with open(pkl_file, 'rb') as f:
            features_dict = pickle.load(f)

        similarity_scores = []

        # Loop through all images in the dataset
        for filename, dataset_features in features_dict.items():
            # Ensure both feature vectors have the same length
            min_length = min(len(query_features), len(dataset_features))
            query_features = query_features[:min_length]
            dataset_features = dataset_features[:min_length]

            # Calculate cosine similarity between query and dataset images
            similarity = cosine_similarity([query_features], [dataset_features])[0][0]
            similarity_scores.append((filename, similarity))

        # Sort images based on similarity (higher similarity first)
        sorted_images = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        return sorted_images[:10]

    def get(self, request):
        return render(request, 'app/edge.html')

    def post(self, request):
        # Get the uploaded query image file
        query_image_file = request.FILES.get('query_image')

        if query_image_file:
            # Create the 'temp' directory if it doesn't exist
            temp_dir = os.path.join(os.getcwd(), 'temp')
            os.makedirs(temp_dir, exist_ok=True)

            # Save the uploaded file to the temporary location
            query_image_path = os.path.join(temp_dir, query_image_file.name)
            with open(query_image_path, 'wb+') as f:
                for chunk in query_image_file.chunks():
                    f.write(chunk)

            # Load the dataset features from the PKL file
            pkl_file = 'app/models/edge.pkl'
            dataset_directory = 'media/database'

            # Call the compare_hog_features function and get the sorted images
            sorted_images = EdgeView.compare_edge_features(query_image_path, dataset_directory, pkl_file)

            # Render the template with the sorted images
            return render(request, 'app/edge.html', {'sorted_images': sorted_images})
        else:
            # Handle the case when no image file is uploaded
            return render(request, 'app/edge.html', {'error_message': 'Please upload an image file.'})


class VggnetView(View):
    http_method_names = ['get', 'post']

    @staticmethod
    def extract_vgg_features(image):
        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
            image_gray = image.squeeze()
        else:
            image_gray = color.rgb2gray(image)

        features = feature.hog(image_gray, orientations=8, pixels_per_cell=(16, 16),
                                cells_per_block=(1, 1), block_norm='L2-Hys',
                                visualize=False, feature_vector=True)
        return features.flatten()

    @staticmethod
    def compare_vgg_features(query_image_path, dataset_directory, pkl_file):
        # Load query image features
        query_image = io.imread(query_image_path)
        query_features = VggnetView.extract_vgg_features(query_image)

        # Load dataset features from the PKL file
        with open(pkl_file, 'rb') as f:
            features_dict = pickle.load(f)

        similarity_scores = []

        # Loop through all images in the dataset
        for filename, dataset_features in features_dict.items():
            # Ensure both feature vectors have the same length
            min_length = min(len(query_features), len(dataset_features))
            query_features = query_features[:min_length]
            dataset_features = dataset_features[:min_length]

            # Calculate cosine similarity between query and dataset images
            similarity = cosine_similarity([query_features], [dataset_features])[0][0]
            similarity_scores.append((filename, similarity))

        # Sort images based on similarity (higher similarity first)
        sorted_images = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        return sorted_images[:10]

    def get(self, request):
        return render(request, 'app/vgg.html')

    def post(self, request):
        # Get the uploaded query image file
        query_image_file = request.FILES.get('query_image')

        if query_image_file:
            # Create the 'temp' directory if it doesn't exist
            temp_dir = os.path.join(os.getcwd(), 'temp')
            os.makedirs(temp_dir, exist_ok=True)

            # Save the uploaded file to the temporary location
            query_image_path = os.path.join(temp_dir, query_image_file.name)
            with open(query_image_path, 'wb+') as f:
                for chunk in query_image_file.chunks():
                    f.write(chunk)

            # Load the dataset features from the PKL file
            pkl_file = 'app/models/vggnet.pkl'
            dataset_directory = 'media/database'

            # Call the compare_hog_features function and get the sorted images
            sorted_images = VggnetView.compare_vgg_features(query_image_path, dataset_directory, pkl_file)

            # Render the template with the sorted images
            return render(request, 'app/vgg.html', {'sorted_images': sorted_images})
        else:
            # Handle the case when no image file is uploaded
            return render(request, 'app/vgg.html', {'error_message': 'Please upload an image file.'})


class ResnetView(View):
    http_method_names = ['get', 'post']

    @staticmethod
    def extract_resnet_features(image):
        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
            image_gray = image.squeeze()
        else:
            image_gray = color.rgb2gray(image)

        features = feature.hog(image_gray, orientations=8, pixels_per_cell=(16, 16),
                                cells_per_block=(1, 1), block_norm='L2-Hys',
                                visualize=False, feature_vector=True)
        return features.flatten()

    @staticmethod
    def compare_resnet_features(query_image_path, dataset_directory, pkl_file):
        # Load query image features
        query_image = io.imread(query_image_path)
        query_features = ResnetView.extract_resnet_features(query_image)

        # Load dataset features from the PKL file
        with open(pkl_file, 'rb') as f:
            features_dict = pickle.load(f)

        similarity_scores = []

        # Loop through all images in the dataset
        for filename, dataset_features in features_dict.items():
            # Ensure both feature vectors have the same length
            min_length = min(len(query_features), len(dataset_features))
            query_features = query_features[:min_length]
            dataset_features = dataset_features[:min_length]

            # Calculate cosine similarity between query and dataset images
            similarity = cosine_similarity([query_features], [dataset_features])[0][0]
            similarity_scores.append((filename, similarity))

        # Sort images based on similarity (higher similarity first)
        sorted_images = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        return sorted_images[:10]

    def get(self, request):
        return render(request, 'app/resnet.html')

    def post(self, request):
        # Get the uploaded query image file
        query_image_file = request.FILES.get('query_image')

        if query_image_file:
            # Create the 'temp' directory if it doesn't exist
            temp_dir = os.path.join(os.getcwd(), 'temp')
            os.makedirs(temp_dir, exist_ok=True)

            # Save the uploaded file to the temporary location
            query_image_path = os.path.join(temp_dir, query_image_file.name)
            with open(query_image_path, 'wb+') as f:
                for chunk in query_image_file.chunks():
                    f.write(chunk)

            # Load the dataset features from the PKL file
            pkl_file = 'app/models/resnet.pkl'
            dataset_directory = 'media/database'

            # Call the compare_hog_features function and get the sorted images
            sorted_images = ResnetView.compare_resnet_features(query_image_path, dataset_directory, pkl_file)

            # Render the template with the sorted images
            return render(request, 'app/resnet.html', {'sorted_images': sorted_images})
        else:
            # Handle the case when no image file is uploaded
            return render(request, 'app/resnet.html', {'error_message': 'Please upload an image file.'})
