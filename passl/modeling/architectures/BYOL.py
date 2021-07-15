# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  -ef |grep main |grep -v grep |cut -c 9-15 |xargs -i kill -9 {}    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import paddle
import paddle.nn as nn
import numpy as np
from ...modules import init
from .builder import MODELS
from ..backbones import build_backbone
from ..necks import build_neck
from ..heads import build_head
import paddle.distributed as dist

# add trnas in network
import paddle
import paddle.fluid.layers as layers

def single_random_gaussian_blur(image, height, width, p=1.0):
    """Randomly blur an image.
    Args:
        image: `Tensor` representing an image of arbitrary size.
        height: Height of output image.
        width: Width of output image.
        p: probability of applying this transformation.
    Returns:
        A preprocessed image `Tensor`.
    """
    image = image.astype("float32")
    kernel_size = height // 10
    padding = 'SAME'
    radius = kernel_size // 2
    kernel_size = radius * 2 + 1
    sigma = paddle.uniform(
        shape=[kernel_size], min=0.1, max=2.0, dtype="float32")
    x = paddle.arange(-radius, radius + 1, 1, "float32")
    blur_filter = paddle.exp(-paddle.pow(x, 2.0) /
                             (2.0 * paddle.pow(sigma, 2.0)))
    blur_filter /= layers.reduce_sum(blur_filter)
    #blur_filter /= blur_filter.sum()
    # One vertical and one horizontal filter.

    #blur_v = layers.reshape(blur_filter, [kernel_size, 1, 1, 1])
    #blur_h = layers.reshape(blur_filter, [1, kernel_size,1, 1])
    blur_v = layers.reshape(blur_filter, [1, 1, kernel_size, 1])
    blur_h = layers.reshape(blur_filter, [1, 1, 1, kernel_size])
    num_channels = 3
    #blur_h = paddle.tile(blur_h, [1, num_channels, 1, 1])
    #blur_v = paddle.tile(blur_v, [1, num_channels, 1, 1])

    blur_h = paddle.tile(blur_h, [num_channels, 1, 1, 1])
    blur_v = paddle.tile(blur_v, [num_channels, 1, 1, 1])
    
    expand_batch_dim = len(image.shape) == 3
    if expand_batch_dim:
        # Tensorflow requires batched input to convolutions, which we can fake with
        # an extra dimension.
        image = paddle.unsqueeze(image.transpose((2,0,1)), axis=0)
    blurred = paddle.nn.functional.conv2d(
        image, blur_h, stride=1, padding=padding,groups=3)
    blurred = paddle.nn.functional.conv2d(
        blurred, blur_v, stride=1, padding=padding,groups=3)
    #if expand_batch_dim:
    #    blurred = paddle.squeeze(blurred, axis=0)
    return blurred.transpose((0,2,3,1))




def random_gaussian_blur(image, height, width, p=1.0):
    """Randomly blur an image.
    Args:
        image: `Tensor` representing an image of arbitrary size.
        height: Height of output image.
        width: Width of output image.
        p: probability of applying this transformation.
    Returns:
        A preprocessed image `Tensor`.
    """
    res = []
    for i in range(image.shape[0]):
        res.append(single_random_gaussian_blur(image[i],height,width,p))
    return paddle.concat(res,axis=0)

def random_solarization(img,threshold=0.5):
    img = paddle.where(img < threshold, img, 1 -img)
    return img

def img_normalize(img,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]):
    mean = paddle.to_tensor(mean, dtype='float32').reshape([1, 1, 1, 3])
    std = paddle.to_tensor(std, dtype='float32').reshape([1, 1, 1, 3])
    return (img - mean) / std

def to_chw(img):
    return img.transpose((0,3,1,2))

def batch_random_blur_solariza_normalize_chw(
                      view1,
                      view2,
                      blur_probability=(1.0,0.1),
                      solariza_probability=(0.0,0.2) ):
    """Apply efficient batch data transformations.
    Args:
        images_list: a list of image tensors.
        height: the height of image.
        width: the width of image.
        blur_probability: the probaility to apply the blur operator.
    Returns:
        Preprocessed feature list.
    """

    def generate_selector(p, bsz):
        shape = [bsz, 1, 1, 1]
        p_tensor = layers.fill_constant(
            shape=shape, dtype="float32", value=p)
        selector = layers.cast(
            layers.less_than(
                layers.uniform_random(
                    shape=shape, min=0, max=1, dtype="float32"),
                p_tensor),
            "float32")
        return selector
 
    B,H,W,C = view1.shape



    #images_list1 = []
    #images_list2 = []
    #for i in range(B):
    img1 = view1
    img1_new = random_gaussian_blur(img1, H, W, p=1.0)
    selector = generate_selector(blur_probability[0],B)
    img1_blur_res = img1_new * selector + img1 * (1 - selector)
    
    selector = generate_selector(solariza_probability[0],B)
    img1_sola_res = random_solarization(img1_blur_res)
    img1_sola_res = img1_sola_res * selector + img1_blur_res * (1 - selector)
    img1_sola_res = paddle.clip(img1_sola_res, min=0., max=1.)
    img1_sola_res.stop_gradient = True

    img1_tran_res = to_chw(img_normalize(img1_sola_res))

    img2 = view2
    img2_new = random_gaussian_blur(img2, H, W, p=1.0)
    selector = generate_selector(blur_probability[1],B)
    img2_blur_res = img2_new * selector + img2 * (1 - selector)
    
    selector = generate_selector(solariza_probability[1],B)
    img2_sola_res = random_solarization(img2_blur_res)
    img2_sola_res = img2_sola_res * selector + img2_blur_res * (1 - selector)
    img2_sola_res = paddle.clip(img2_sola_res, min=0., max=1.)
    img2_sola_res.stop_gradient = True    

    img2_tran_res = to_chw(img_normalize(img2_sola_res))
    

    return img1_tran_res, img2_tran_res

@MODELS.register()
class BYOL(nn.Layer):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 predictor=None,
                 dim=256,
                 num_classes=1000,
                 embedding_dim=2048,
                 target_decay_method='fixed',
                 target_decay_rate=0.996,
                 align_init_network=True,
                 use_synch_bn=False
                ):
        """
        Args:
            backbone (dict): config of backbone.
            neck (dict): config of neck.
            head (dict): config of head.
            dim (int): feature dimension. Default: 256.
        """
        super(BYOL, self).__init__()
        # create the encoders
        # num_classes is the output fc dimension
        self.towers = nn.LayerList()
        self.base_m = target_decay_rate
        self.target_decay_method = target_decay_method
        
        neck1 = build_neck(neck)
        neck2 = build_neck(neck)
        
        self.towers.append(nn.Sequential(build_backbone(backbone), neck1))
        self.towers.append(nn.Sequential(build_backbone(backbone), neck2))
        self.net_init(self.towers)
        self.predictor = build_neck(predictor)
        self.net_init(self.predictor)
        self.classifier = nn.Linear(embedding_dim,num_classes)
        self.net_init(self.classifier)

        self.backbone = self.towers[0][0]
        # self.neck1 = self.towers[0][1]

        # TODO IMPORTANT! Explore if the initialization requires to be synchronized
        for param_q, param_k in zip(self.towers[0].parameters(),self.towers[1].parameters()):
            param_k.stop_gradient = True

        if align_init_network:
            for param_q, param_k in zip(self.towers[0].parameters(),self.towers[1].parameters()):
                param_k.set_value(param_q)  # initialize
                
        # Convert BatchNorm*d to SyncBatchNorm*d
        if use_synch_bn:
            self.towers[0] = nn.SyncBatchNorm.convert_sync_batchnorm(self.towers[0])
            self.towers[1] = nn.SyncBatchNorm.convert_sync_batchnorm(self.towers[1])
            #self.predictor = nn.SyncBatchNorm.convert_sync_batchnorm(self.predictor)

        self.head = build_head(head)
    
    def net_init(self,network):
        for m in network.sublayers():
            if isinstance(m, nn.Conv2D):
                init.kaiming_init(m,mode="fan_in",nonlinearity="conv2d")
            if isinstance(m, nn.Conv2D):
                init.kaiming_init(m,mode="fan_in",nonlinearity="linear")

    def train_iter(self, *inputs, **kwargs):
        
        current_iter = kwargs['current_iter']
        total_iters =  kwargs['total_iters']
        
        if self.target_decay_method == 'cosine':
            self.m = 1 - (1-self.base_m) * (1 + math.cos(math.pi*(current_iter-1)/total_iters))/2.0   # 47.0
        elif self.target_decay_method == 'fixed':
            self.m = self.base_m   # 55.7
        else:
            raise NotImplementedError

        # self.update_target_network()
        img_a, img_b, label = inputs
        img_a, img_b = batch_random_blur_solariza_normalize_chw(img_a,img_b)
       
        '''a1 = self.predictor(self.towers[0](img_a))
        b1 = self.towers[1](img_b)

        a1 = nn.functional.normalize(a1, axis=1)
        b1 = nn.functional.normalize(b1, axis=1)
        b1.stop_gradient = True

        a2 = self.predictor(self.towers[0](img_b))
        b2 = self.towers[1](img_a)

        a2 = nn.functional.normalize(a2, axis=1)
        b2 = nn.functional.normalize(b2, axis=1)
        b2.stop_gradient = True
        ''' 
        embedding = self.towers[0][0](img_a)
        online_project_view1 = self.towers[0][1](embedding)
        online_predict_view1 = self.predictor(online_project_view1)

        online_project_view2 = self.towers[0](img_b)
        online_predict_view2 = self.predictor(online_project_view2)
        
        clone_x = embedding.clone()
        clone_x.stop_gradient = True 
        classif_out = self.classifier(clone_x.squeeze())
        
        with paddle.no_grad():
            target_project_view1 = self.towers[1](img_a).clone().detach()
            target_project_view2 = self.towers[1](img_b).clone().detach()


        a1 = nn.functional.normalize(online_predict_view1, axis=1)
        b1 = nn.functional.normalize(target_project_view2, axis=1)
        b1.stop_gradient = True        

        a2 = nn.functional.normalize(online_predict_view2, axis=1)
        b2 = nn.functional.normalize(target_project_view1, axis=1)
        b2.stop_gradient = True

        outputs = self.head(a1, b1, a2, b2, classif_out, label)

        return outputs

    def forward(self, *inputs, mode='train', **kwargs):
        if mode == 'train':
            return self.train_iter(*inputs, **kwargs)
        elif mode == 'test':
            return self.test_iter(*inputs, **kwargs)
        elif mode == 'extract':
            return self.backbone(*inputs)
        else:
            raise Exception("No such mode: {}".format(mode))

    # original EMA
    @paddle.no_grad()
    def update_target_network(self):
        for param_q, param_k in zip(self.towers[0].parameters(),
                                    self.towers[1].parameters()):
            paddle.assign((param_k * self.m + param_q * (1. - self.m)), param_k)
            #param_k.set_value(param_k * self.m + param_q * (1. - self.m))
            param_k.stop_gradient = True

    # L1 update
    @paddle.no_grad()
    def update_target_network_L1(self):
        for param_q, param_k in zip(self.towers[0].parameters(),
                                    self.towers[1].parameters()):
            paddle.assign(param_k - (1-self.m)*paddle.sign(param_k-param_q), param_k)
            param_k.stop_gradient = True

    # L2 + L1
    @paddle.no_grad()
    def update_target_network_clip(self):
        for param_q, param_k in zip(self.towers[0].parameters(),
                                    self.towers[1].parameters()):
            # paddle.assign((param_k * self.m + param_q * (1. - self.m)), param_k)
            paddle.assign(param_k - (1-self.m) * paddle.clip((param_k - param_q), min=-1.0, max=1.0) , param_k)
            param_k.stop_gradient = True

    @paddle.no_grad()
    def update_target_network_LN_clip(self):
        for param_q, param_k in zip(self.towers[0].parameters(),
                                    self.towers[1].parameters()):
            paddle.assign((param_k * self.m + param_q * (1. - self.m)), param_k)
            paddle.assign(param_k - (1-self.m) * paddle.clip((param_k - param_q), min=-1.0, max=1.0) , param_k)
            param_k.stop_gradient = True
