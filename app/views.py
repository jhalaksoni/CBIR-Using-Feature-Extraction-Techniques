import os
import cv2
import pafy
import math
import pickle
import random
import string
import base64
import logging
import tempfile
import numpy as np
from ultralytics import YOLO
from django.views import View
from django.urls import reverse
from django.conf import settings
from django.core.files import File
from django.contrib import messages
from app.models import ProcessedVideo
from skimage import io, color, feature
from sklearn.preprocessing import normalize
from django.contrib.messages import get_messages
from django.core.files.temp import NamedTemporaryFile
from sklearn.metrics.pairwise import cosine_similarity
from django.shortcuts import render,redirect,HttpResponseRedirect
from skimage.feature import graycomatrix, graycoprops , match_descriptors
from django.http import JsonResponse, FileResponse, StreamingHttpResponse, HttpResponseBadRequest, HttpResponseServerError


logger = logging.getLogger(__name__)



from skimage.filters import gabor_kernel
from scipy.ndimage import convolve
from scipy.spatial.distance import euclidean


import os
import numpy as np
import imageio.v2 as imageio
from math import sqrt
from django.shortcuts import render
from django.views import View
from django.conf import settings
from django.core.files.storage import default_storage
from six.moves import cPickle

# Edge Detection Class
class Edge(object):
    edge_kernels = np.array([
        [[1, -1], [1, -1]],
        [[1, 1], [-1, -1]],
        [[sqrt(2), 0], [0, -sqrt(2)]],
        [[0, sqrt(2)], [-sqrt(2), 0]],
        [[2, -2], [-2, 2]]
    ])
    
    def histogram(self, input, stride=(2, 2), type='region', n_slice=7, normalize=True):
        if isinstance(input, np.ndarray):
            img = input.copy()
        else:
            img = imageio.imread(input)
        height, width, channel = img.shape

        if type == 'global':
            hist = self._conv(img, stride=stride, kernels=self.edge_kernels)
        elif type == 'region':
            hist = np.zeros((n_slice, n_slice, self.edge_kernels.shape[0]))
            h_silce = np.around(np.linspace(0, height, n_slice+1, endpoint=True)).astype(int)
            w_slice = np.around(np.linspace(0, width, n_slice+1, endpoint=True)).astype(int)
            
            for hs in range(len(h_silce)-1):
                for ws in range(len(w_slice)-1):
                    img_r = img[h_silce[hs]:h_silce[hs+1], w_slice[ws]:w_slice[ws+1]]
                    hist[hs][ws] = self._conv(img_r, stride=stride, kernels=self.edge_kernels)
        
        if normalize:
            hist /= np.sum(hist)
        
        return hist.flatten()

    def _conv(self, img, stride, kernels, normalize=True):
        H, W, C = img.shape
        conv_kernels = np.expand_dims(kernels, axis=3)
        conv_kernels = np.tile(conv_kernels, (1, 1, 1, C))
        
        sh, sw = stride
        kn, kh, kw, kc = conv_kernels.shape
        hh = int((H - kh) / sh + 1)
        ww = int((W - kw) / sw + 1)
        
        hist = np.zeros(kn)
        
        for idx, k in enumerate(conv_kernels):
            for h in range(hh):
                hs = int(h * sh)
                he = int(h * sh + kh)
                for w in range(ww):
                    ws = w * sw
                    we = w * sw + kw
                    hist[idx] += np.sum(img[hs:he, ws:we] * k)
        
        if normalize:
            hist /= np.sum(hist)
        
        return hist

# Django View
class EdgeDetectionView(View):
    http_method_names = ['get', 'post']
    
    @staticmethod
    def compare_edge_features(query_image_path, dataset_directory, pkl_file):
        query_image = imageio.imread(query_image_path)
        edge_detector = Edge()
        query_features = edge_detector.histogram(query_image)
        
        with open(pkl_file, 'rb') as f:
            features_dict = cPickle.load(f)
        
        similarity_scores = []
        
        for filename, dataset_features in features_dict.items():
            min_length = min(len(query_features), len(dataset_features))
            query_features = query_features[:min_length]
            dataset_features = dataset_features[:min_length]
            
            similarity = np.dot(query_features, dataset_features) / (np.linalg.norm(query_features) * np.linalg.norm(dataset_features))
            similarity_scores.append((filename, similarity))
        
        sorted_images = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        return sorted_images[:10]
    
    def get(self, request):
        return render(request, 'app/edge_detection.html')
    
    def post(self, request):
        query_image_file = request.FILES.get('query_image')
        
        if query_image_file:
            temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            
            query_image_path = os.path.join(temp_dir, query_image_file.name)
            with default_storage.open(query_image_path, 'wb+') as f:
                for chunk in query_image_file.chunks():
                    f.write(chunk)
            
            pkl_file = os.path.join(settings.MEDIA_ROOT, 'extracted_features_edge.pkl')
            dataset_directory = os.path.join(settings.MEDIA_ROOT, 'outfit')
            
            sorted_images = self.compare_edge_features(query_image_path, dataset_directory, pkl_file)
            
            return render(request, 'app/edge_detection.html', {'sorted_images': sorted_images})
        else:
            return render(request, 'app/edge_detection.html', {'error_message': 'Please upload an image file.'})


class GaborView(View):
    http_method_names = ['get', 'post']

    @staticmethod
    def extract_gabor_features(image, kernel_params):
        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
            image_gray = image.squeeze()
        else:
            image_gray = color.rgb2gray(image)

        features = []
        for params in kernel_params:
            orientation = params['orientation']
            if orientation == 0:
                theta = 0
            else:
                theta = np.pi / orientation

            kernel = np.real(gabor_kernel(params['frequency'], theta=theta))
            filtered = convolve(image_gray, kernel, mode='nearest')
            features.append(filtered.ravel())

        return np.concatenate(features)

    @staticmethod
    def compare_images(query_image_path, pkl_file, kernel_params):
        # Load query image features
        query_image = io.imread(query_image_path)
        query_features = GaborView.extract_gabor_features(query_image, kernel_params)

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


        # Sort images based on similarity (lower similarity first)
        sorted_images = sorted(similarity_scores, key=lambda x: x[1])

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
            pkl_file = 'app/models/extracted_features_gabor.pkl'

            # Example kernel parameters
            kernel_params = [
                {'frequency': 0.1, 'orientation': 1},
                {'frequency': 0.2, 'orientation': np.pi/4},
                {'frequency': 0.3, 'orientation': np.pi/2},
                {'frequency': 0.4, 'orientation': 3*np.pi/4}
            ]

            # Call the compare_images function and get the sorted images
            sorted_images = GaborView.compare_images(query_image_path, pkl_file, kernel_params)

            # Render the template with the sorted images
            return render(request, 'app/gabor.html', {'sorted_images': sorted_images})
        else:
            # Handle the case when no image file is uploaded
            return render(request, 'app/gabor.html', {'error_message': 'Please upload an image file.'})

class GLCMView(View):
    http_method_names = ['get', 'post']

    @staticmethod
    def extract_glcm_features(image, distances=[1], angles=[0], levels=256, props=('contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation')):
        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
            image_gray = image.squeeze()
        else:
            image_gray = color.rgb2gray(image)

        # Normalize the image to [0, levels-1] range
        image_gray = (image_gray * (levels - 1) / image_gray.max()).astype(np.uint8)
        features = []
        for distance in distances:
            for angle in angles:
                glcm = graycomatrix(image_gray, [distance], [angle], levels=levels)
                for prop in props:
                    features.append(graycoprops(glcm, prop)[0, 0])
        return np.array(features)
        # return features.flatten()


    @staticmethod
    def compare_glcm_features(query_image_path, pkl_file):
        # Load query image features
        query_image = io.imread(query_image_path)
        query_features = GLCMView.extract_glcm_features(query_image)

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
        return sorted_images
    def get(self, request):
        return render(request, 'app/glcm.html')

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
            pkl_file = 'app/models/extracted_features_glcm.pkl'

            # Call the compare_glcm_features function and get the sorted images
            sorted_images = GLCMView.compare_glcm_features(query_image_path, pkl_file)

            # Render the template with the sorted images
            return render(request, 'app/glcm.html', {'sorted_images': sorted_images})
        else:
            # Handle the case when no image file is uploaded
            return render(request, 'app/glcm.html', {'error_message': 'Please upload an image file.'})
        
class SurfView(View):
    http_method_names = ['get', 'post']

    @staticmethod
    def extract_surf_features(image):
        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
            gray = image.squeeze()
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        return descriptors


    @staticmethod
    def compare_surf_features(query_image_path, dataset_directory, pkl_file):
        # Load query image features
        query_image = io.imread(query_image_path)
        query_features = SurfView.extract_surf_features(query_image)

        # Load dataset features from the PKL file
        with open(pkl_file, 'rb') as f:
            features_dict = pickle.load(f)

        similarity_scores = []

        # Loop through all images in the dataset
        for filename, dataset_features in features_dict.items():
            # Ensure both feature vectors have the same length
        

            if dataset_features is not None and query_features is not None:
                # Calculate the number of matching features
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(query_features, dataset_features, k=2)
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

                # Calculate similarity score based on the number of good matches
                similarity = len(good_matches) / max(len(query_features), len(dataset_features))
                similarity_scores.append((filename, similarity))

        # Sort images based on similarity (higher similarity first)
        sorted_images = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        return sorted_images

    def get(self, request):
        return render(request, 'app/surf.html')

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
            pkl_file = 'app/models/extracted_features_surf.pkl'
            dataset_directory = 'media/outfit'

            # Call the compare_hog_features function and get the sorted images
            sorted_images = SurfView.compare_surf_features(query_image_path, dataset_directory, pkl_file)

            # Render the template with the sorted images
            return render(request, 'app/surf.html', {'sorted_images': sorted_images})
        else:
            # Handle the case when no image file is uploaded
            return render(request, 'app/surf.html', {'error_message': 'Please upload an image file.'})
        
        
class SiftView(View):
    http_method_names = ['get', 'post']

    @staticmethod
    def extract_sift_features(image):
        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
            image_gray = image.squeeze()
        else:
            image_gray = color.rgb2gray(image)
        
        image8bit = cv2.normalize(image_gray, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(image8bit, None)
        return des

    @staticmethod
    def compare_sift_features(query_image_path, dataset_directory, pkl_file):
        # Load query image features
        query_image = io.imread(query_image_path)
        query_features = SiftView.extract_sift_features(query_image)

        # Load dataset features from the PKL file
        with open(pkl_file, 'rb') as f:
            features_dict = pickle.load(f)

        similarity_scores = []

        # Loop through all images in the dataset
        for filename, dataset_features in features_dict.items():
            if query_features is not None and dataset_features is not None:
            # Ensure both feature vectors have the same length
                min_length = min(len(query_features), len(dataset_features))
                query_features_truncated = query_features[:min_length]
                dataset_features_truncated = dataset_features[:min_length]

                # Calculate cosine similarity between query and dataset images
                similarity = np.mean(cosine_similarity(dataset_features_truncated, query_features_truncated))
                similarity_scores.append((filename, similarity))

        # Sort images based on similarity (higher similarity first)
        sorted_images = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        return sorted_images[:10]

    def get(self, request):
        return render(request, 'app/sift.html')

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
            pkl_file = 'app/models/extracted_features_sift.pkl'
            dataset_directory = 'media/outfit'

            # Call the compare_hog_features function and get the sorted images
            sorted_images = SiftView.compare_sift_features(query_image_path, dataset_directory, pkl_file)

            # Render the template with the sorted images
            return render(request, 'app/sift.html', {'sorted_images': sorted_images})
        else:
            # Handle the case when no image file is uploaded
            return render(request, 'app/sift.html', {'error_message': 'Please upload an image file.'})


class LensView(View):
    http_method_names = ['get', 'post']

    @staticmethod
    def extract_hog_features(image):
        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
            image_gray = image.squeeze()
        else:
            image_gray = color.rgb2gray(image)

        features = feature.hog(image_gray, orientations=8, pixels_per_cell=(16, 16),
                                cells_per_block=(1, 1), block_norm='L2-Hys',
                                visualize=False, feature_vector=True)
        return features.flatten()

    @staticmethod
    def compare_hog_features(query_image_path, dataset_directory, pkl_file):
        # Load query image features
        query_image = io.imread(query_image_path)
        query_features = LensView.extract_hog_features(query_image)

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
            pkl_file = 'app/models/extracted_features_hog.pkl'
            dataset_directory = 'media/outfit'

            # Call the compare_hog_features function and get the sorted images
            sorted_images = LensView.compare_hog_features(query_image_path, dataset_directory, pkl_file)

            # Render the template with the sorted images
            return render(request, 'app/lens.html', {'sorted_images': sorted_images})
        else:
            # Handle the case when no image file is uploaded
            return render(request, 'app/lens.html', {'error_message': 'Please upload an image file.'})




# Create your views here.
class CBIRView(View):
  def get(self,request):
      
      return render(
        request,
        'app/home.html',
        {
            
        }
      )
