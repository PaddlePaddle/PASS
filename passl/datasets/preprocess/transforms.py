# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import random
import numpy as np
import paddle
import cv2

from paddle.vision.transforms import  RandomResizedCrop,Transpose,Resize,CenterCrop,Normalize,RandomHorizontalFlip
from .builder import TRANSFORMS, build_transform
from .cv2_trans import ByolRandomHorizontalFlip, ByolColorJitter, ByolRandomGrayscale, ByolNormalize,ToCHW,ToRGB,ByolCenterCrop

TRANSFORMS.register(RandomResizedCrop)
TRANSFORMS.register(ByolRandomHorizontalFlip)
TRANSFORMS.register(ByolColorJitter)
TRANSFORMS.register(ByolRandomGrayscale)
TRANSFORMS.register(ByolNormalize)
TRANSFORMS.register(ToCHW)
TRANSFORMS.register(ToRGB)

TRANSFORMS.register(Transpose)
TRANSFORMS.register(Resize)
TRANSFORMS.register(CenterCrop)
TRANSFORMS.register(ByolCenterCrop)
TRANSFORMS.register(Normalize)
TRANSFORMS.register(RandomHorizontalFlip)

@TRANSFORMS.register()
class To_Normal():
    def __init__(self,mean,std):
        self.mean=mean
        self.std=std
    def __call__(self,img):
        self.mean = np.array(self.mean, dtype='float32').reshape([1, 1, 3])
        self.std = np.array(self.std, dtype='float32').reshape([1, 1, 3])
        return (img - self.mean) /self.std    #.astype("float32")

@TRANSFORMS.register()
class NormToOne():
    def __init__(self):
        pass

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Randomly grayscaled image.
        """
        #norm_img = (img/255.).astype('float32')
        norm_img = Image.eval(img,lambda x:x/255.0)
        return norm_img
@TRANSFORMS.register()
class Clip():
    def __init__(self,min_val=0.0,max_val=1.0):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Cliped image.
        """
        clip_img = img.clip(self.min_val,self.max_val)
        return clip_img

@TRANSFORMS.register()
class RandomApply():
    """Apply randomly a list of transformations with a given probability

    Args:
        transforms (list or tuple): list of transformations
        p (float): probability
    """

    def __init__(self, transforms, p=0.5):
        _transforms = []
        if isinstance(transforms, (list, tuple)):
            for transform in transforms:
                if isinstance(transform, dict):
                    _transforms.append(build_transform(transform))
                else:
                    _transforms.append(transform)

        self.transforms = _transforms
        self.p = p

    def __call__(self, img):
        if self.p < random.random():
            return img
        for t in self.transforms:
            img = t(img)
        return img




@TRANSFORMS.register()
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self,kernel_size, sigma=[.1, 2.],use_cv=True):
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.use_cv = use_cv

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        if self.use_cv:
            x = x[:,:,::-1]  # toBGR
            x = cv2.GaussianBlur(x, (self.kernel_size, self.kernel_size), sigma)
            x = x[:,:,::-1]  # toRGB
            return x
        else:
            x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
            return x


@TRANSFORMS.register()
class Solarization(object):
    """Solarization augmentation in BYOL https://arxiv.org/abs/2006.07733."""

    def __init__(self, threshold=128,norm=False):
        self.threshold = threshold
        self.norm = norm

    def __call__(self, img):
        #print(img.shape,"  =====> info:",img.max(),img.min(),img.mean(),img.std())
        if not self.norm:
            img = np.where(img < self.threshold, img, 255 -img)
        else:
            img = np.where(img < self.threshold, img, 1 -img)
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
