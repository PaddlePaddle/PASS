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

import os
import pickle
import paddle

from .hook import Hook
from .builder import HOOKS


def save(state_dicts, file_name):
    def convert(state_dict):
        model_dict = {}

        for k, v in state_dict.items():
            if isinstance(
                    v,
                (paddle.fluid.framework.Variable, paddle.fluid.core.VarBase)):
                model_dict[k] = v.numpy()
            else:
                model_dict[k] = v

        return model_dict

    final_dict = {}
    for k, v in state_dicts.items():
        if isinstance(
                v,
            (paddle.fluid.framework.Variable, paddle.fluid.core.VarBase)):
            final_dict = convert(state_dicts)
            break
        elif isinstance(v, dict):
            final_dict[k] = convert(v)
        else:
            final_dict[k] = v

    with open(file_name, 'wb') as f:
        pickle.dump(final_dict, f)


@HOOKS.register()
class CheckpointHook(Hook):
    """Save checkpoints periodically.

    Args:
        interval (int): The saving period. If ``by_epoch=True``, interval
            indicates epochs, otherwise it indicates iterations.
            Default: -1, which means "never".
        by_epoch (bool): Saving checkpoints by epoch or by iteration.
            Default: True.
        save_optimizer (bool): Whether to save optimizer state_dict in the
            checkpoint. It is usually used for resuming experiments.
            Default: True.
        out_dir (str, optional): The directory to save checkpoints. If not
            specified, ``trainer.work_dir`` will be used by default.
        max_keep_ckpts (int, optional): The maximum checkpoints to keep.
            In some cases we want only the latest few checkpoints and would
            like to delete old ones to save the disk space.
            Default: -1, which means unlimited.
    """
    def __init__(self,
                 interval=1,
                 by_epoch=True,
                 save_optimizer=True,
                 out_dir=None,
                 max_keep_ckpts=5,
                 priority=1,
                 **kwargs):
        self.priority = priority
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.max_keep_ckpts = max_keep_ckpts
        self.out_dir = out_dir
        self.args = kwargs

    def save_checkpoint(self,
                        out_dir,
                        trainer,
                        filename_tmpl='epoch_{}.pdparams',
                        save_optimizer=True,
                        create_symlink=False):
        filename = filename_tmpl.format(trainer.current_epoch + 1)
        filepath = os.path.join(out_dir, filename)
        optimizer = trainer.optimizer if save_optimizer else None
        lr_scheduler = trainer.lr_scheduler
        save(
            {
                'epoch': trainer.current_epoch + 1,
                'state_dict': trainer.model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_sheduer': lr_scheduler.state_dict()
            }, filepath)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            os.symlink(filename, os.path.join(out_dir, 'latest.pth'))

    def train_epoch_end(self, trainer):
        if paddle.distributed.get_rank() != 0:
            return

        if not self.by_epoch or not self.every_n_epochs(trainer, self.interval):
            return

        trainer.logger.info(
            f'Saving checkpoint at {trainer.current_epoch + 1} epochs')
        if not self.out_dir:
            self.out_dir = trainer.output_dir
        self.save_checkpoint(self.out_dir,
                             trainer,
                             save_optimizer=self.save_optimizer,
                             **self.args)

        # remove other checkpoints
        if self.max_keep_ckpts > 0:
            filename_tmpl = self.args.get('filename_tmpl', 'epoch_{}.pdparams')
            current_epoch = trainer.current_epoch + 1
            for epoch in range(current_epoch - self.max_keep_ckpts, 0, -1):
                ckpt_path = os.path.join(self.out_dir,
                                         filename_tmpl.format(epoch))
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
                else:
                    print(ckpt_path)
                    break

    def run_end(self, trainer):
        if paddle.distributed.get_rank() != 0:
            return

        if not self.by_epoch or not self.every_n_epochs(trainer, self.interval):
            return

        trainer.logger.info(
            f'Saving checkpoint at {trainer.current_epoch + 2} epochs')
        if not self.out_dir:
            self.out_dir = trainer.output_dir
        self.save_checkpoint(self.out_dir,
                             trainer,
                             save_optimizer=self.save_optimizer,
                             **self.args)



    def train_iter_end(self, trainer):
        if paddle.distributed.get_rank() != 0:
            return

        if self.by_epoch or not self.every_n_iters(trainer, self.interval):
            return

        trainer.logger.info(
            f'Saving checkpoint at {trainer.iter + 1} iterations')
        if not self.out_dir:
            self.out_dir = trainer.output_dir
        trainer.save_checkpoint(self.out_dir,
                                save_optimizer=self.save_optimizer,
                                **self.args)

        # remove other checkpoints
        if self.max_keep_ckpts > 0:
            filename_tmpl = self.args.get('filename_tmpl', 'iter_{}.pdparams')
            current_iter = trainer.iter + 1
            for _iter in range(
                    current_iter - self.max_keep_ckpts * self.interval, 0,
                    -self.interval):
                ckpt_path = os.path.join(self.out_dir,
                                         filename_tmpl.format(_iter))
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
                else:
                    break
