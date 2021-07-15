# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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


from __future__ import absolute_import

import cv2
import random
import numpy as np
import types
from PIL import Image
from paddle.vision.transforms import CenterCrop

from . import cv2_func as F


_cv2_str_to_interpolation = {
    'nearest': cv2.INTER_NEAREST,
    'linear': cv2.INTER_LINEAR,
    'area': cv2.INTER_AREA,
    'cubic': cv2.INTER_CUBIC,
    'lanczos4': cv2.INTER_LANCZOS4,
}


class Compose(object):
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, img):
        for t in self.trans:
            img = t(img)
        return img


class ByolNormalize(object):
    def __init__(self, mean=None, std=None):
        if mean is None or std is None:
            self.mean = None
            self.std = None
        else:
            # HWC
            self.mean = np.array(mean, dtype='float32').reshape([1, 1, 3])
            self.std = np.array(std, dtype='float32').reshape([1, 1, 3])

    def __call__(self, img):
        img = np.array(img)
        return F.normalize(img, self.mean, self.std)


class Resize(object):
    """ size, int or (h, w)"""
    def __init__(self, size, interpolation='linear'):
        assert isinstance(size, int) or len(size) == 2
        self.size = size
        self.interpolation = _cv2_str_to_interpolation[interpolation]

    def __call__(self, img):
        return F.resize(img, self.size, self.interpolation)

'''
class CenterCrop(object):
    """ size, int or (h, w)"""
    def __init__(self, size):
        if type(size) is int:
            self.size = (size, size)
        else:
            self.size = size  # (h, w)

    def __call__(self, img):
        return F.center_crop(img, self.size)
'''

class ToCHW(object):
    def __call__(self, img):
        return F.to_chw(img)


class ToRGB(object):
    def __call__(self, img):
        return F.to_rgb_bgr(img)


class Lambda(object):
    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)


class RandomTransforms(object):
    def __init__(self, trans):
        assert isinstance(trans, (list, tuple))
        self.trans = trans

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class RandomApply(RandomTransforms):
    def __init__(self, trans, p=0.5):
        super(RandomApply, self).__init__(trans)
        self.p = p

    def __call__(self, img):
        if self.p < random.random():
            return img
        for t in self.trans:
            img = t(img)
        return img


class ByolRandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return F.hflip(img)
        return img


class ByolRandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if np.random.uniform() <= self.p:
            return F.vflip(img)
        return img


class ByolRandomResizedCrop(object):
    def __init__(self,
                 size,
                 scale=(0.08, 1.0),
                 ratio=(3./4., 4./3.),
                 interpolation='linear'):
        if type(size) is int:
            self.size = (size, size)
        else:
            self.size = size  # (h, w)
        self.scale = scale
        self.ratio = ratio
        self.interpolation = _cv2_str_to_interpolation[interpolation]

    def __call__(self, img):
        return F.random_crop_with_resize(
            img, self.size, self.scale, self.ratio, self.interpolation)


class ByolColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = [- brightness, brightness]
        self.contrast = [1 - contrast, 1 + contrast]
        self.saturation = [1 - saturation, 1 + saturation]
        self.hue = [-hue, hue]

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        transforms = []

        brightness_factor = np.random.uniform(brightness[0], brightness[1])
        contrast_factor = np.random.uniform(contrast[0], contrast[1])
        saturation_factor = np.random.uniform(saturation[0], saturation[1])
        hue_factor = np.random.uniform(hue[0], hue[1])

        transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))
        transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))
        transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))
        transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)
        return transform

    def __call__(self, img):
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img)


class ByolRandomGrayscale(object):
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img):
        if np.random.uniform() <= self.p:
            return F.to_grayscale(img)
        return img

class ByolCenterCrop(object):
    def __init__(self):
        pass
    
    def __call__(self,img):
        image_width, image_height = img.size
        padded_center_crop_size = int((224 / (224 + 32)) *np.minimum(image_height, image_width))
        return CenterCrop(size=padded_center_crop_size)(img)
