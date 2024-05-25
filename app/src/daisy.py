# -*- coding: utf-8 -*-

from __future__ import print_function

from .evaluate import evaluate_class
from .DB import Database

from skimage.feature import daisy
from skimage import color

from six.moves import cPickle
import numpy as np
import imageio.v2  # Replace imageio.v2 with imageio.v2
import math

import os

# cache dir
cache_dir = 'cache'
if not os.path.exists(cache_dir):
  os.makedirs(cache_dir)


class Daisy(object):
  def __init__(self):
    self.n_slice    = 2
    self.n_orient   = 8
    self.step       = 4  # Reduced from 10
    self.radius     = 10  # Reduced from 30
    self.rings      = 2
    self.histograms = 6
    self.h_type     = 'region'
    self.d_type     = 'd1'

    self.depth      = 3

    self.R = (self.rings * self.histograms + 1) * self.n_orient

  def histogram(self, input, type=None, n_slice=None, normalize=True):
    ''' count img histogram
  
      arguments
        input    : a path to a image or a numpy.ndarray
        type     : 'global' means count the histogram for whole image
                   'region' means count the histogram for regions in images, then concatanate all of them
        n_slice  : work when type equals to 'region', height & width will equally sliced into N slices
        normalize: normalize output histogram
  
      return
        type == 'global'
          a numpy array with size self.R
        type == 'region'
          a numpy array with size n_slice * n_slice * self.R
    '''
    type = type or self.h_type
    n_slice = n_slice or self.n_slice

    if isinstance(input, np.ndarray):  # examine input type
      img = input.copy()
    else:
      img = imageio.v2.imread(input, mode='RGB')
    height, width, channel = img.shape

    if type == 'global':
      hist = self._daisy(img)
  
    elif type == 'region':
      hist = np.zeros((n_slice, n_slice, self.R))
      h_silce = np.around(np.linspace(0, height, n_slice + 1, endpoint=True)).astype(int)
      w_slice = np.around(np.linspace(0, width, n_slice + 1, endpoint=True)).astype(int)
  
      for hs in range(len(h_silce) - 1):
        for ws in range(len(w_slice) - 1):
          img_r = img[h_silce[hs]:h_silce[hs + 1], w_slice[ws]:w_slice[ws + 1]]  # slice img to regions
          hist[hs][ws] = self._daisy(img_r)
  
    if normalize:
      hist /= np.sum(hist)
  
    return hist.flatten()
  
  def _daisy(self, img, normalize=True):
    image = color.rgb2gray(img)
    descs = daisy(image, step=self.step, radius=self.radius, rings=self.rings, histograms=self.histograms, orientations=self.n_orient)
    descs = descs.reshape(-1, self.R)  # shape=(N, self.R)
    hist  = np.mean(descs, axis=0)  # shape=(self.R,)
  
    if normalize:
      hist = np.array(hist) / np.sum(hist)
  
    return hist
  
  def make_samples(self, db, verbose=True):
    if self.h_type == 'global':
      sample_cache = "daisy-{}-n_orient{}-step{}-radius{}-rings{}-histograms{}".format(self.h_type, self.n_orient, self.step, self.radius, self.rings, self.histograms)
    elif self.h_type == 'region':
      sample_cache = "daisy-{}-n_slice{}-n_orient{}-step{}-radius{}-rings{}-histograms{}".format(self.h_type, self.n_slice, self.n_orient, self.step, self.radius, self.rings, self.histograms)
  
    try:
      samples = cPickle.load(open(os.path.join(cache_dir, sample_cache), "rb"))
      for sample in samples:
        sample['hist'] /= np.sum(sample['hist'])  # normalize
      if verbose:
        print("Using cache..., config=%s, distance=%s, depth=%s" % (sample_cache, self.d_type, self.depth))
    except:
      if verbose:
        print("Counting histogram..., config=%s, distance=%s, depth=%s" % (sample_cache, self.d_type, self.depth))
  
      samples = []
      data = db.get_data()
      for d in data.itertuples():
        d_img, d_cls = getattr(d, "img"), getattr(d, "cls")
        d_hist = self.histogram(d_img, type=self.h_type, n_slice=self.n_slice)
        samples.append({
                        'img':  d_img, 
                        'cls':  d_cls, 
                        'hist': d_hist
                      })
      cPickle.dump(samples, open(os.path.join(cache_dir, sample_cache), "wb"))
  
    return samples

if __name__ == "__main__":
  db = Database()

  # evaluate database
  APs = evaluate_class(db, f_class=Daisy, d_type=d_type, depth=depth)
  cls_MAPs = []
  for cls, cls_APs in APs.items():
    MAP = np.mean(cls_APs)
    print("Class {}, MAP {}".format(cls, MAP))
    cls_MAPs.append(MAP)
  print("MMAP", np.mean(cls_MAPs))
