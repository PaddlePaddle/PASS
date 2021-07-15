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

import copy
import numpy as np
import paddle
from .preprocess import build_transforms
from ..utils.registry import Registry, build_from_config

DATASETS = Registry("DATASET")
def build_dataset(cfg):
    return build_from_config(cfg, DATASETS)


def build_dataloader(cfg):
    cfg_ = copy.deepcopy(cfg)
    dataset_cfg = cfg_.pop('dataset')
    sampler_cfg = cfg_.pop('sampler')
    
    dataset = build_dataset(dataset_cfg)
    sampler = paddle.io.DistributedBatchSampler(dataset, **sampler_cfg)

    dataloader = paddle.io.DataLoader(dataset, batch_sampler=sampler, **cfg_)

    return dataloader
