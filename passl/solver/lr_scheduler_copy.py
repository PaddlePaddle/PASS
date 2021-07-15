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

import math
import paddle

from paddle.optimizer.lr import LRScheduler, MultiStepDecay, CosineAnnealingDecay, LinearWarmup
from .builder import LRSCHEDULERS, build_lr_scheduler

class Cosinlr(LRScheduler):
    def __init__(self,T_max=100,learning_rate=0.1, last_epoch=-1, verbose=False):
        self.T_max = T_max
        super(Cosinlr, self).__init__(learning_rate,last_epoch,verbose)

    def get_lr(self):
        return self.base_lr * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2

class CosinWarmup(LinearWarmup):
    def __init__(self,total_image,total_batch,total_steps,warmup_steps,start_lr,end_lr,last_epoch=-1,verbose=False):
        total_steps = total_steps * total_image // total_batch
        warmup_steps = warmup_steps * total_image // total_batch
        T_max = total_steps - warmup_steps
        lr = Cosinlr(T_max,end_lr,last_epoch=-1)
        super(CosinWarmup, self).__init__(lr,warmup_steps,start_lr,end_lr,last_epoch=-1,verbose=False)



LRSCHEDULERS.register(MultiStepDecay)
LRSCHEDULERS.register(CosineAnnealingDecay)
LRSCHEDULERS.register(LinearWarmup)
LRSCHEDULERS.register(LRScheduler)
LRSCHEDULERS.register(Cosinlr)
LRSCHEDULERS.register(CosinWarmup)
