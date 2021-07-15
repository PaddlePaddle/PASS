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

import paddle
import paddle.nn as nn

from .builder import HEADS
from ...modules.init import reset_parameters, normal_init


@HEADS.register()
class ClasHead(nn.Layer):
    """Simple classifier head.
    """
    def __init__(self, with_avg_pool=False, in_channels=2048, num_classes=1000):
        super(ClasHead, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.criterion = nn.CrossEntropyLoss()

        if self.with_avg_pool:
            self.avg_pool = nn.AdaptiveAvgPool2D((1, 1))
        self.fc_cls = nn.Linear(in_channels, num_classes)
        # reset_parameters(self.fc_cls)
        normal_init(self.fc_cls, mean=0.0, std=0.01, bias=0.0)

    def forward(self, x):
        if self.with_avg_pool:
            assert x.dim() == 4, \
                "Tensor must has 4 dims, got: {}".format(x.dim())
            x = self.avg_pool(x)
        x = x.reshape([x.shape[0], -1])
        cls_score = self.fc_cls(x)
        return cls_score

    def loss(self, cls_score, labels):
        losses = dict()

        losses['loss'] = self.criterion(cls_score, labels)
        losses['final_loss'] = losses['loss']
        losses['acc1'], losses['acc5'] = accuracy(cls_score,
                                                  labels,
                                                  topk=(1, 5))
        return losses


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with paddle.no_grad():
        maxk = max(topk)
        batch_size = target.shape[0]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = paddle.cast(pred == target.reshape([1, -1]).expand_as(pred),
                              'float32')

        res = []
        for k in topk:
            correct_k = correct[:k].reshape([-1]).sum(0, keepdim=True)
            res.append(correct_k * 100.0 / batch_size)
        return res
