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

import datetime
import os.path as osp

from .hook import Hook
from .builder import HOOKS
from ..utils import AverageMeter


@HOOKS.register()
class LogHook(Hook):
    """Simple logger hook."""

    def __init__(self,
                 by_epoch=True,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True,
                 priority=1):

        self.interval = interval
        self.ignore_last = ignore_last
        self.reset_flag = reset_flag

        self.by_epoch = by_epoch
        self.time_sec_tot = 0
        self.priority = priority

    def run_begin(self, trainer):
        self.start_iter = trainer.current_iter
        self.json_log_path = osp.join(trainer.output_dir,
                                      f'{trainer.timestamp}.log.json')

    def _log_info(self, log_dict, trainer):

        if trainer.mode == 'train':
            if isinstance(log_dict['lr'], dict):
                lr_str = []
                for k, val in log_dict['lr'].items():
                    lr_str.append(f'lr_{k}: {val:.3e}')
                lr_str = ' '.join(lr_str)
            else:
                lr_str = f'lr: {log_dict["lr"]:.3e}'

            # by epoch: Epoch [4/100][100/1000]
            # by iter:  Iter [100/100000]
            if self.by_epoch:
                log_str = f'Epoch [{log_dict["epoch"]}/{trainer.epochs}]' \
                          f'[{log_dict["iter"]}/{trainer.iters_per_epoch}]\t'
            else:
                log_str = f'Iter [{log_dict["iter"]}/{trainer.total_iters}]\t'
            log_str += f'{lr_str}, '

            if 'time' in log_dict.keys():
                self.time_sec_tot += (log_dict['time'].sum)
                time_sec_avg = log_dict['time'].avg
                eta_sec = time_sec_avg * (
                    trainer.total_iters - trainer.current_iter - 1)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                log_str += f'eta: {eta_str}, '
                log_str += f'time: {time_sec_avg:.3f}, ' \
                           f'data_time: {log_dict["data_time"].avg:.3f}, '

        else:
            if self.by_epoch:
                log_str = f'Epoch({log_dict["mode"]}) ' \
                        f'[{log_dict["epoch"] - 1}][{log_dict["iter"]}]\t'
            else:
                log_str = f'Iter({log_dict["mode"]}) [{log_dict["iter"]}]\t'

        log_items = []
        for name, val in log_dict.items():
            if name in [
                    'mode', 'Epoch', 'iter', 'lr', 'time', 'data_time',
                    'memory', 'epoch'
            ]:
                continue
            if isinstance(val, AverageMeter):
                val = str(val)
            log_items.append(val)

        log_str += ', '.join(log_items)

        trainer.logger.info(log_str)

    def _round_float(self, items):
        if isinstance(items, list):
            return [self._round_float(item) for item in items]
        elif isinstance(items, float):
            return round(items, 5)
        else:
            return items

    def print_log(self, trainer):
        log_dict = trainer.logs
        # training mode if the output contains the key "time"
        mode = 'train' if 'time' in trainer.logs else 'val'
        log_dict['mode'] = mode
        log_dict['epoch'] = trainer.current_epoch + 1
        if self.by_epoch:
            log_dict['iter'] = trainer.inner_iter + 1
        else:
            log_dict['iter'] = trainer.current_iter + 1

        cur_lr = trainer.lr_scheduler[0].get_lr()
        if isinstance(cur_lr, list):
            log_dict['lr'] = cur_lr[0]
        elif isinstance(cur_lr, dict):

            log_dict['lr'] = {}
            for k, lr_ in cur_lr.items():
                assert isinstance(lr_, list)
                log_dict['lr'].update({k: lr_[0]})
        else:
            log_dict['lr'] = cur_lr

        if mode == 'train':
            log_dict['time'] = trainer.logs['time']
            log_dict['data_time'] = trainer.logs['data_time']

        for name, val in trainer.logs.items():
            if name in ['time', 'data_time']:
                continue
            log_dict[name] = val

        self._log_info(log_dict, trainer)

    def epoch_begin(self, trainer):
        trainer.logs.clear()

    def train_iter_end(self, trainer):
        for k, v in trainer.outputs.items():
            if k not in trainer.logs:
                if 'loss' in k:
                    fmt = ':.4e'
                else:
                    fmt = ':6.3f'
                trainer.logs[k] = AverageMeter(k, fmt)
            try:
                trainer.logs[k].update(float(v))
            except Exception as e:
                print(e,"   ",k,"   ",v)

        if self.by_epoch and self.every_n_inner_iters(trainer, self.interval):
            self.print_log(trainer)

    def train_epoch_end(self, trainer):
        if self.reset_flag:
            trainer.logs.clear()

    def val_epoch_end(self, trainer):
        self.print_log(trainer)
        if self.reset_flag:
            trainer.logs.clear()
