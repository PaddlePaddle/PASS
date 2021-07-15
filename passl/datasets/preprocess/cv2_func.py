# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains image preprocessing methods."""
import cv2
import math
import random
import numpy as np


def to_rgb_bgr(img):
    """ convert BRG->RGB, or RGB->BGR
    Args:
        img: HWC(bgr or rgb)
    Returns:
        converted image.
    """
    return img[:, :, ::-1]


def to_chw(img):
    """ HWC to CHW """
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img.transpose((2, 0, 1))
    return img


def normalize(img, mean=None, std=None):
    """ Normalize img with HWC(rgb) in the range[0, 255]
        img = img / 255.0
        img = (img - mean) / std
    Args:
        img: HWC(rgb) in the range [0, 255]
        mean: means for each channel, if None,
        std:
    Returns:
        normalize img HWC(rgb).
    """
    # to HWC(rgb) in the range[0.0, 1.0]
    img = img / np.float32(255.0)

    if mean is None or std is None:
        return img

    return (img - mean) / std


def hflip(img):
    # return cv2.flip(img, 1)
    return img[:, ::-1, :]


def vflip(img):
    # return cv2.flip(img, 0)
    return img[::-1, :, :]


def adjust_brightness(img, factor): 
    return (img + factor).clip(0,1) 


def adjust_contrast(img, factor):
    def _adjust_contrast_channel(channel):
        mean = np.mean(channel, axis=(-2, -1), keepdims=True)
        return (factor * (channel - mean) + mean).clip(0,1)
    return _adjust_contrast_channel(img)


def rgb_to_hsv(img):
  """Converts R, G, B  values to H, S, V values.
  Reference TF implementation:
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/adjust_saturation_op.cc
  Only input values between 0 and 1 are guaranteed to work properly, but this
  function complies with the TF implementation outside of this range.
  Args:
    r: A tensor representing the red color component as floats.
    g: A tensor representing the green color component as floats.
    b: A tensor representing the blue color component as floats.
  Returns:
    H, S, V values, each as tensors of shape [...] (same as the input without
    the last dimension).
  """
  r, g, b = img[:,:,0],img[:,:,1],img[:,:,2]
  vv = np.maximum(np.maximum(r, g), b)
  range_ = vv - np.minimum(np.minimum(r, g), b)
  sat = np.where(vv > 0, range_ / vv, 0.)

  norm = np.where(range_ != 0, 1. / (6. * range_), 1e9)

  hr = norm * (g - b)
  hg = norm * (b - r) + 2. / 6.
  hb = norm * (r - g) + 4. / 6.

  hue = np.where(r == vv, hr, np.where(g == vv, hg, hb))
  hue = hue * (range_ > 0)
  hue = hue + (hue < 0)

  return hue, sat, vv

def hsv_to_rgb(h, s, v):
  """Converts H, S, V values to an R, G, B tuple.
  Reference TF implementation:
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/adjust_saturation_op.cc
  Only input values between 0 and 1 are guaranteed to work properly, but this
  function complies with the TF implementation outside of this range.
  Args:
    h: A float tensor of arbitrary shape for the hue (0-1 values).
    s: A float tensor of the same shape for the saturation (0-1 values).
    v: A float tensor of the same shape for the value channel (0-1 values).
  Returns:
    An (r, g, b) tuple, each with the same dimension as the inputs.
  """
  c = s * v
  m = v - c
  dh = (h % 1.) * 6.
  fmodu = dh % 2.
  x = c * (1 - np.abs(fmodu - 1))
  hcat = np.floor(dh).astype(np.int32)
  rr = np.where(
      (hcat == 0) | (hcat == 5), c, np.where(
          (hcat == 1) | (hcat == 4), x, 0)) + m
  gg = np.where(
      (hcat == 1) | (hcat == 2), c, np.where(
          (hcat == 0) | (hcat == 3), x, 0)) + m
  bb = np.where(
      (hcat == 3) | (hcat == 4), c, np.where(
          (hcat == 2) | (hcat == 5), x, 0)) + m
  rr = np.expand_dims(rr, axis=-1)
  gg = np.expand_dims(gg, axis=-1)
  bb = np.expand_dims(bb, axis=-1)
  rgb_img = np.concatenate((rr, gg, bb),axis=-1)
  return rgb_img

def adjust_saturation(img, factor):
    h,s,v = rgb_to_hsv(img)
    s = (s * factor).clip(0,1)
    return hsv_to_rgb(h,s,v).clip(0,1)


def adjust_hue(img, factor):
    h,s,v = rgb_to_hsv(img)
    h = (h + factor) % 1.0
    return hsv_to_rgb(h,s,v).clip(0,1)


def to_grayscale(image):
    rgb_weights = np.array([0.2989, 0.5870, 0.1140])
    grayscale = np.tensordot(image, rgb_weights, axes=(-1, -1))[..., np.newaxis]
    return np.tile(grayscale, (1, 1, 3)).clip(0,1)  # Back to 3 channels.

def resize(img, size, interpolation=cv2.INTER_LINEAR):
    """ resize """
    if isinstance(size, int):
        h, w = img.shape[:2]  # hwc

        min_hw = min(h, w)
        if size == min_hw:
            return img

        percent = float(size) / min_hw
        rw = int(round(w * percent))
        rh = int(round(h * percent))
        out = cv2.resize(img, (rw, rh), interpolation=interpolation)
    else:
        # size (h, w)
        out = cv2.resize(img, (size[1], size[0]), interpolation=interpolation)

    return out


def crop(img, h_start, w_start, h, w):
    h_end = h_start + h
    w_end = w_start + w
    return img[h_start:h_end, w_start:w_end, :]


def center_crop(img, size):
    if type(size) is int:
        size = (size, size)

    h, w = size
    img_h, img_w = img.shape[:2]
    w_start = int(round(img_w - w) / 2.)
    h_start = int(round(img_h - h) / 2.)

    return crop(img, h_start, w_start, h, w)


def random_crop_with_resize(img,
                            size,
                            scale=(0.08, 1.0),
                            ratio=(3./4., 4./3.),
                            interpolation=cv2.INTER_LINEAR):
    """ random_crop and resized to given size"""
    if type(size) is int:
        size = (size, size)
    aspect_ratio = math.sqrt(random.uniform(*ratio))
    w = 1. * aspect_ratio
    h = 1. / aspect_ratio

    bound = min((float(img.shape[1]) / img.shape[0]) / (w ** 2),
                (float(img.shape[0]) / img.shape[1]) / (h ** 2))
    scale_max = min(scale[1], bound)
    scale_min = min(scale[0], bound)

    target_area = img.shape[0] * img.shape[1] * random.uniform(scale_min, scale_max)
    target_size = math.sqrt(target_area)
    w = int(target_size * w)
    h = int(target_size * h)

    i = random.randint(0, img.shape[0] - h)
    j = random.randint(0, img.shape[1] - w)

    img = img[i:i+h, j:j+w, :]
    resized = cv2.resize(img, dsize=(size[1], size[0]),
                         interpolation=interpolation)
    return resized


# ====================================


def rotate_image(img):
    """ rotate_image """
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    angle = random.randint(-10, 10)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated


def crop_image(img, size, center):
    """ crop_image """
    height, width = img.shape[:2]
    if center == True:
        w_start = (width - size) / 2
        h_start = (height - size) / 2
    else:
        w_start = random.randint(0, width - size)
        h_start = random.randint(0, height - size)
    w_end = w_start + size
    h_end = h_start + size
    img = img[h_start:h_end, w_start:w_end, :]
    return img

def gaussian_blur(img,blur_divider,sigma):
     kernel_size = images.shape[1] / blur_divider
     radius = int(kernel_size / 2)
     kernel_size_ = 2 * radius + 1
     x = np.arange(-radius, radius + 1).astype(np.float32)
     blur_filter = np.exp(-x**2 / (2. * sigma**2))
     blur_filter = blur_filter / jnp.sum(blur_filter)
     blur_v = jnp.reshape(blur_filter, [kernel_size_, 1, 1, 1])
     blur_h = jnp.reshape(blur_filter, [1, kernel_size_, 1, 1])
     num_channels = image.shape[-1]
     blur_h = jnp.tile(blur_h, [1, 1, 1, num_channels])
     blur_v = jnp.tile(blur_v, [1, 1, 1, num_channels])
     expand_batch_dim = len(image.shape) == 3
     if expand_batch_dim:
         image = image[jnp.newaxis, ...]
     blurred = _depthwise_conv2d(image, blur_h, strides=[1, 1], padding=padding)
     blurred = _depthwise_conv2d(blurred, blur_v, strides=[1, 1], padding=padding)
     blurred = jnp.squeeze(blurred, axis=0)
     return blurred
