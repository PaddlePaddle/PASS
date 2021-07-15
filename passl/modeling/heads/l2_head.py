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


@HEADS.register()
class L2Head(nn.Layer):
    """Head for contrastive learning.

    Args:
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Default: 0.1.
    """

    def __init__(self):
        super(L2Head, self).__init__()
        self.class_criteria = nn.CrossEntropyLoss()

    def forward(self, x1, y1, x2, y2,classif_out, label):
        outputs = dict()
        loss1 = ((x1 - y1)**2).sum(axis=1)
        loss2 = ((x2 - y2)**2).sum(axis=1)
        outputs['rep1'] = loss1.mean()
        outputs['rep2'] = loss2.mean()
        outputs['rep_loss'] = (loss1 + loss2).mean()
        
        class_loss = self.class_criteria(classif_out,label)
        outputs["class_loss"] = class_loss.mean()
        outputs["final_loss"] = outputs['rep_loss'] + outputs["class_loss"]
        return outputs
