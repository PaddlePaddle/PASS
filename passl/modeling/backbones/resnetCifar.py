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
import paddle.vision.models as models
from paddle.vision.models.resnet import BasicBlock, BottleneckBlock

from .builder import BACKBONES
from ...modules import init, freeze
from ...utils.logger import get_logger

class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x


@BACKBONES.register()
class ResNetCifar(models.ResNet):
    """ResNet model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        Block (BasicBlock|BottleneckBlock): block module of model.
        depth (int): layers of resnet, default: 50.
        num_classes (int): output dim of last fc layer. If num_classes <=0, last fc layer
                            will not be defined. Default: 1000.
        with_pool (bool): use pool before the last fc layer or not. Default: True.

    Examples:
        .. code-block:: python

            from paddle.vision.models import ResNet
            from paddle.vision.models.resnet import BottleneckBlock, BasicBlock

            resnet50 = ResNet(BottleneckBlock, 50)

            resnet18 = ResNet(BasicBlock, 18)

    """

    def __init__(self,
                 depth,
                 num_classes=0,
                 with_pool=False,
                 zero_init_residual=False,
                 frozen_stages=-1,
                 freeze_bn=True,
                 pretrained=None):

        block = BasicBlock if depth in [18, 34] else BottleneckBlock

        super(ResNetCifar, self).__init__(block, depth, num_classes, with_pool)
        self.conv1 = nn.Conv2D(
                        3,
                        64,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias_attr=False)
        self.maxpool = Identity()

        self.zero_init_residual = zero_init_residual
        self.frozen_stages = frozen_stages
        self.init_parameters()
        if pretrained is not None:
            state_dict = paddle.load(pretrained)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            self.set_state_dict(state_dict)
            logger = get_logger()
            logger.info('Load pretrained backbone weight from {} success!'.format(pretrained))

        self.freeze_bn = freeze_bn
        self._freeze_stages()

    def init_parameters(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                init.kaiming_init(m, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.layer.norm._BatchNormBase, nn.GroupNorm)):
                init.constant_init(m, 1)

        if self.zero_init_residual:
            for m in self.sublayers():
                if isinstance(m, BottleneckBlock):
                    init.constant_init(m.bn3, 0)
                elif isinstance(m, BasicBlock):
                    init.constant_init(m.bn2, 0)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.freeze_bn:
                freeze.freeze_batchnorm_statictis(self.bn1)
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.trainable = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            if self.freeze_bn:
                freeze.freeze_batchnorm_statictis(m)
            for param in m.parameters():
                param.trainable = False

        if self.frozen_stages >= 0:
            logger = get_logger()
            logger.info(
                'Frozen layer before stage {}'.format(self.frozen_stages + 1))