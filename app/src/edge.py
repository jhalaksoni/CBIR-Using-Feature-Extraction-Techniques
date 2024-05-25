from skimage import io, color
import numpy as np
from math import sqrt
import os
from six.moves import cPickle

class Edge(object):
  def __init__(self):
      self.stride = (1, 1)
      self.n_slice = 7
      self.h_type = 'region'
      self.d_type = 'd1'
      self.depth = 4
      self.edge_kernels = np.array([
          [[1, -1], [1, -1]],  # vertical
          [[1, 1], [-1, -1]],  # horizontal
          [[sqrt(2), 0], [0, -sqrt(2)]],  # 45 diagonal
          [[0, sqrt(2)], [-sqrt(2), 0]],  # 135 diagonal
          [[2, -2], [-2, 2]]  # non-directional
      ])

  def histogram(self, input, stride=(2, 2), type='region', n_slice=7, normalize=True):
      if isinstance(input, np.ndarray):  # examine input type
          img = input.copy()
      else:
          img = io.imread(input)
          if img.ndim == 2:  # Grayscale to RGB
              img = color.gray2rgb(img)
          elif img.shape[2] == 4:  # Remove alpha channel if present
              img = img[:, :, :3]
      height, width, channel = img.shape

      if type == 'global':
          hist = self._conv(img, stride=stride, kernels=self.edge_kernels)
      elif type == 'region':
          hist = np.zeros((n_slice, n_slice, self.edge_kernels.shape[0]))
          h_slice = np.around(np.linspace(0, height, n_slice + 1, endpoint=True)).astype(int)
          w_slice = np.around(np.linspace(0, width, n_slice + 1, endpoint=True)).astype(int)

          for hs in range(len(h_slice) - 1):
              for ws in range(len(w_slice) - 1):
                  img_r = img[h_slice[hs]:h_slice[hs + 1], w_slice[ws]:w_slice[ws + 1]]  # slice img to regions
                  hist[hs][ws] = self._conv(img_r, stride=stride, kernels=self.edge_kernels)

      if normalize:
          hist /= np.sum(hist)

      return hist.flatten()

  def _conv(self, img, stride, kernels, normalize=True):
      H, W, C = img.shape
      conv_kernels = np.expand_dims(kernels, axis=3)
      conv_kernels = np.tile(conv_kernels, (1, 1, 1, C))
      assert list(conv_kernels.shape) == list(kernels.shape) + [C]  # check kernels size

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
                  hist[idx] += np.sum(img[hs:he, ws:we] * k)  # element-wise product

      if normalize:
          hist /= np.sum(hist)

      return hist


  
  def make_samples(self, db, verbose=True):
    if h_type == 'global':
      sample_cache = "edge-{}-stride{}".format(h_type, stride)
    elif h_type == 'region':
      sample_cache = "edge-{}-stride{}-n_slice{}".format(h_type, stride, n_slice)
  
    try:
      samples = cPickle.load(open(os.path.join(cache_dir, sample_cache), "rb"))
      for sample in samples:
        sample['hist'] /= np.sum(sample['hist'])  # normalize
      if verbose:
        print("Using cache..., config=%s, distance=%s, depth=%s" % (sample_cache, d_type, depth))
    except:
      if verbose:
        print("Counting histogram..., config=%s, distance=%s, depth=%s" % (sample_cache, d_type, depth))
  
      samples = []
      data = db.get_data()
      for d in data.itertuples():
        d_img, d_cls = getattr(d, "img"), getattr(d, "cls")
        d_hist = self.histogram(d_img, type=h_type, n_slice=n_slice)
        samples.append({
                        'img':  d_img, 
                        'cls':  d_cls, 
                        'hist': d_hist
                      })
      cPickle.dump(samples, open(os.path.join(cache_dir, sample_cache), "wb"))
  
    return samples


if __name__ == "__main__":
  db = Database()

  # check shape
  assert edge_kernels.shape == (5, 2, 2)

  # evaluate database
  APs = evaluate_class(db, f_class=Edge, d_type=d_type, depth=depth)
  cls_MAPs = []
  for cls, cls_APs in APs.items():
    MAP = np.mean(cls_APs)
    print("Class {}, MAP {}".format(cls, MAP))
    cls_MAPs.append(MAP)
  print("MMAP", np.mean(cls_MAPs))
