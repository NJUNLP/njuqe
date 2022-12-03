# Copyright (c) 2020 the NJUQE authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import Collection
from dataclasses import dataclass, field
from typing import List

from fairseq.dataclass import FairseqDataclass
from omegaconf import II, DictConfig

from fairseq.optim.lr_scheduler import FairseqLRScheduler, register_lr_scheduler


@dataclass
class NoamScheduleConfig(FairseqDataclass):
    warmup_updates: int = field(
        default=4000,
        metadata={"help": "warmup the learning rate linearly for the first N updates"},
    )
    embed_dim: float = field(
        default=512,
        metadata={
            "help": "initial learning rate during warmup phase; default is args.lr"
        },
    )
    # TODO common vars at parent class
    lr: List[float] = II("optimization.lr")


@register_lr_scheduler("noam", dataclass=NoamScheduleConfig)
class NoamSchedule(FairseqLRScheduler):
    """
    """

    def __init__(self, cfg: DictConfig, optimizer):
        super().__init__(cfg, optimizer)
        if isinstance(cfg.lr, Collection) and len(cfg.lr) > 1:
            raise ValueError(
                "Cannot use a fixed learning rate schedule with noam."
                " Consider --lr-scheduler=fixed instead."
            )

        # initial learning rate
        self.embed_dim = cfg.embed_dim
        self.init_lr = (
            cfg.lr[0]
            if isinstance(cfg.lr, Collection)
            else cfg.lr
        )
        self.lr = self.init_lr
        self.optimizer.set_lr(self.lr)
        self.warmup_updates = cfg.warmup_updates   # 从cfg中初始化

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        num_updates += 1  # 因为默认从0开始，下面的负开跟操作会报错
        opt_corr = 0.002
        origin_lr = self.init_lr * self.embed_dim ** (-0.5) * opt_corr * 5000.0
        self.lr = origin_lr * min(num_updates ** (-0.5),
                                  num_updates * self.warmup_updates ** (-1.5))
        self.optimizer.set_lr(self.lr)
        return self.lr
