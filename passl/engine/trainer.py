#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import logging
from tqdm import tqdm
from collections import OrderedDict

import paddle
import paddle.distributed as dist

from ..hooks import build_hook, Hook
from ..utils.misc import AverageMeter
from ..modules import DistributedDataParallel
from ..datasets.builder import build_dataloader
from ..modeling.architectures import build_model
from ..solver import build_lr_scheduler, build_optimizer, MultiStateDictMeta


class IterLoader:
    def __init__(self, dataloader, epoch=0):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)


class Trainer:
    r"""
    # trainer calling logic:
    #
    #                build_model                               ||    model(BaseModel)
    #                     |                                    ||
    #               build_dataloader                           ||    dataloader
    #                     |                                    ||
    #               build_lr_scheduler                         ||    lr_scheduler
    #                     |                                    ||
    #               build_optimizer                            ||    optimizers
    #                     |                                    ||
    #               build_train_hooks                          ||    train hooks
    #                     |                                    ||
    #               build_custom_hooks                         ||    custom hooks
    #                     |                                    ||
    #                 train loop                               ||    train loop
    #                     |                                    ||
    #      hook(print log, checkpoint, evaluate, ajust lr)     ||    call hook
    #                     |                                    ||
    #                    end                                   \/
    """
    def __init__(self, cfg):
        # base config
        self.logger = logging.getLogger(__name__)
        self.cfg = cfg
        self.output_dir = cfg.output_dir

        self.local_rank = dist.get_rank() 

        self.start_epoch = 0
        self.current_epoch = 0
        self.current_iter = 0
        self.inner_iter = 0
        self.batch_id = 0
        self.global_steps = 0
        self.timestamp = cfg.timestamp
        self.logs = OrderedDict()

        # build model
        self.model = build_model(cfg.model)
        # multiple gpus prepare
        if dist.get_world_size() > 1:
            paddle.distributed.init_parallel_env()
            self.model = DistributedDataParallel(self.model)

        # build train dataloader
        self.train_dataloader  = build_dataloader(cfg.dataloader.train)
        self.iters_per_epoch = len(self.train_dataloader)
        print("whc______data:",len(self.train_dataloader))

        
        # build optimizer
        self.lr_scheduler = MultiStateDictMeta()
        self.optimizer = MultiStateDictMeta()
        # if type(parameters) == list: 
        '''
        if not hasattr(self.model._layers, 'separate_parameters'):
            parameters = self.model.parameters()
            # build lr scheduler
            self.lr_scheduler.append(build_lr_scheduler(cfg.lr_scheduler, self.iters_per_epoch))
            self.optimizer.append(build_optimizer(cfg.optimizer, self.lr_scheduler[0], parameters))
        else:
            print('Using Seperating Learning Rate')
            parameters = self.model._layers.separate_parameters()
            for key, value in parameters.items():
                current_lr_scheduler = build_lr_scheduler(getattr(cfg.lr_scheduler, key), self.iters_per_epoch)
                self.lr_scheduler.append(current_lr_scheduler)
                self.optimizer.append(build_optimizer(cfg.optimizer, current_lr_scheduler, value))
        print("self.lr_scheduler:   ",self.lr_scheduler)
        print(self.lr_scheduler.state_dict())
        '''
        # build hooks
        self.hooks = []

        self.add_train_hooks()
        self.add_custom_hooks()
        self.hooks = sorted(self.hooks, key=lambda x: x.priority)

        self.epochs = cfg.get('epochs', None)
        if self.epochs:
            self.total_iters = self.epochs * self.iters_per_epoch
            total_batch = cfg.get('total_batch', None)
            total_images = cfg.get('total_image', None)
            if total_batch is not None and total_images is not None:
                self.total_iters = self.epochs * total_images // total_batch
            self.by_epoch = True
        else:
            self.by_epoch = False
            self.total_iters = cfg.total_iters

    def add_train_hooks(self):
        optim_cfg = self.cfg.get('optimizer_config', None)
        if optim_cfg is not None:
            self.add_hook(optim_cfg)
        else:
            self.add_hook(build_hook({'name': 'OptimizerHook'}))

        lr_cfg = self.cfg.get('lr_config', None)
        if lr_cfg is not None:
            self.add_hook(lr_cfg)
        else:
            self.add_hook(build_hook({'name': 'LRSchedulerHook'}))

        timer_cfg = self.cfg.get('timer_config', None)
        if timer_cfg is not None:
            self.add_hook(timer_cfg)
        else:
            self.add_hook(build_hook({'name': 'IterTimerHook'}))

        ckpt_cfg = self.cfg.get('checkpoint', None)
        if ckpt_cfg is not None:
            self.add_hook(ckpt_cfg)
        else:
            self.add_hook(build_hook({'name': 'CheckpointHook'}))

        log_cfg = self.cfg.get('log_config', None)
        if log_cfg is not None:
            self.add_hook(build_hook(log_cfg))
        else:
            self.add_hook(build_hook({'name': 'LogHook'}))
        
        # visual config
        #visual_cfg = self.cfg.get('visual_config', None) 
        #if visual_cfg is None:
        #    self.add_hook(build_hook(visual_cfg))
        #else:
        #    self.add_hook(build_hook({'name': 'VisualHook'}))

    def add_custom_hooks(self):
        custom_cfgs = self.cfg.get('custom_config', None)
        if custom_cfgs is None:
            return

        for custom_cfg in custom_cfgs:
            cfg_ = custom_cfg.copy()
            insert_index = cfg_.pop('insert_index', None)
            self.add_hook(build_hook(cfg_), insert_index)

    def add_hook(self, hook, insert_index=None):
        assert isinstance(hook, Hook)

        if insert_index is None:
            self.hooks.append(hook)
        elif isinstance(insert_index, int):
            self.hooks.insert(insert_index, hook)

    def call_hook(self, fn_name):
        for hook in self.hooks:
            getattr(hook, fn_name)(self)

    def train(self):
        self.mode = 'train'
        self.model.train()
        iter_loader = IterLoader(self.train_dataloader, self.current_epoch)
        self.call_hook('run_begin')

        while self.current_iter < (self.total_iters):
            if self.current_iter % self.iters_per_epoch == 0:
                self.call_hook('train_epoch_begin')

            self.current_iter += 1
            self.current_epoch = iter_loader.epoch
            self.inner_iter = self.current_iter % self.iters_per_epoch

            data = next(iter_loader)
            self.call_hook('train_iter_begin')

            self.outputs = self.model(*data, total_iters=self.total_iters, current_iter=self.current_iter)
            
            self.call_hook('train_iter_end')

            if self.current_iter % self.iters_per_epoch == 0:
                self.call_hook('train_epoch_end')
                self.current_epoch += 1

        self.call_hook('run_end')

    def val(self, **kargs):
        if not hasattr(self, 'val_dataloader'):
            self.val_dataloader = build_dataloader(self.cfg.dataloader.val)

        self.logger.info(
            'start evaluate on epoch {} ..'.format(self.current_epoch + 1))
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        model = self.model
        total_samples = len(self.val_dataloader.dataset)
        self.logger.info('Evaluate total samples {}'.format(total_samples))

        if rank == 0:
            dataloader = tqdm(self.val_dataloader)
        else:
            dataloader = self.val_dataloader

        accum_samples = 0

        self.model.eval()
        outs = OrderedDict()

        for data in dataloader:
            if isinstance(data, paddle.Tensor):
                batch_size = data.shape[0]
            elif isinstance(data, (list, tuple)):
                batch_size = data[0].shape[0]
            else:
                raise TypeError('unknown type of data')

            labels = data[-1]
            pred = model(*data, mode='test')

            current_samples = batch_size * world_size
            accum_samples += current_samples

            # for k, v in outputs.items():
            if world_size > 1:
                pred_list = []
                dist.all_gather(pred_list, pred)
                pred = paddle.concat(pred_list, 0)
                label_list = []
                dist.all_gather(label_list, labels)
                labels = paddle.concat(label_list, 0)
                if accum_samples > total_samples:
                    self.logger.info('total samples {} {} {}'.format(
                        total_samples, accum_samples,
                        total_samples + current_samples - accum_samples))
                    pred = pred[:total_samples + current_samples -
                                accum_samples, ...]
                    labels = labels[:total_samples + current_samples -
                                    accum_samples, ...]
                    current_samples = total_samples + current_samples - accum_samples

            res = self.val_dataloader.dataset.evaluate(pred, labels, **kargs)

            for k, v in res.items():
                if k not in outs:
                    outs[k] = AverageMeter(k, ':6.3f')
                outs[k].update(float(v), current_samples)

        log_str = f'Validate Epoch [{self.current_epoch + 1}] '
        log_items = []
        for name, val in outs.items():
            if isinstance(val, AverageMeter):
                string = '{} ({' + outs[k].fmt + '})'
                val = string.format(val.name, val.avg)

            log_items.append(val)
        log_str += ', '.join(log_items)
        self.logger.info(log_str)

        self.model.train()

    def resume(self, checkpoint_path):
        checkpoint = paddle.load(checkpoint_path)
        if checkpoint.get('epoch', None) is not None:
            self.start_epoch = checkpoint['epoch']
            self.current_epoch = checkpoint['epoch']
            self.current_iter = (self.start_epoch - 0) * self.iters_per_epoch
        
        self.model.set_state_dict(checkpoint['state_dict'])
        self.optimizer.set_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.set_state_dict(checkpoint['lr_sheduer'])
        self.logger.info(
            'Resume training from {} success!'.format(checkpoint_path))

    def load(self, weight_path):
        state_dict = paddle.load(weight_path)

        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        self.model.set_state_dict(state_dict)
