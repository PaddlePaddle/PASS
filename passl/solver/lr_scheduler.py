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
class CosinLinearWarmup(LRScheduler):
    def __init__(self,
                 learning_rate,
                 T_max,
                 warmup_steps,
                 start_lr,
                 end_lr,
                 last_epoch=-1,
                 verbose=False):
        type_check = isinstance(learning_rate, float) or isinstance(
            learning_rate, int) or isinstance(learning_rate, LRScheduler)
        if not type_check:
            raise TypeError(
                "the type of learning_rate should be [int, float or LRScheduler], the current type is {}".
                    format(learning_rate))
        self.learning_rate = learning_rate
        self.T_max = T_max
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.end_lr = end_lr
        assert end_lr > start_lr, "end_lr {} must be greater than start_lr {}".format(
            end_lr, start_lr)
        super(CosinLinearWarmup, self).__init__(start_lr, last_epoch, verbose)

    def state_dict(self):
        """
        Returns the state of the LinearWarmup scheduler as a :class:`dict`.

        It is a subset of ``self.__dict__`` .
        """
        state_dict = super(CosinLinearWarmup, self).state_dict()
        if isinstance(self.learning_rate, LRScheduler):
            state_dict["LinearWarmup_LR"] = self.learning_rate.state_dict()
        return state_dict

    def set_state_dict(self, state_dict):
        """
        Loads state_dict for LinearWarmup scheduler.
        """
        super(CosinLinearWarmup, self).set_state_dict(state_dict)
        if isinstance(self.learning_rate, LRScheduler):
            self.learning_rate.set_state_dict(state_dict["LinearWarmup_LR"])

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return (self.end_lr - self.start_lr) * float(
                self.last_epoch) / float(self.warmup_steps) + self.start_lr
        else:
            if isinstance(self.learning_rate, LRScheduler):
                lr_value = self.learning_rate()
                self.learning_rate.step()
                return lr_value
            return self.learning_rate * 0.5 * (1 + math.cos(math.pi * (self.last_epoch-self.warmup_steps) / self.T_max))




class CosinWarmup(CosinLinearWarmup):
    def __init__(self,total_image,total_batch,total_steps,warmup_steps,start_lr,end_lr,last_epoch=-1,verbose=False):
        total_steps = total_steps * total_image // total_batch
        warmup_steps = warmup_steps * total_image // total_batch
        T_max = total_steps - warmup_steps
        super(CosinWarmup, self).__init__(end_lr,T_max,warmup_steps,start_lr,end_lr,last_epoch=-1,verbose=False)



LRSCHEDULERS.register(MultiStepDecay)
LRSCHEDULERS.register(CosineAnnealingDecay)
LRSCHEDULERS.register(LinearWarmup)
LRSCHEDULERS.register(LRScheduler)
LRSCHEDULERS.register(CosinWarmup)
