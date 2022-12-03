# Copyright (c) 2020 the NJUQE authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fairseq.logging.meters import Meter


class LabelMeter(Meter):
    """Stores target and predicted label"""

    def __init__(self):
        self.y = None
        self.y_hat = None
        self.reset()

    def reset(self):
        self.y = None
        self.y_hat = None

    def update(self, labels):
        y = labels[0]
        y_hat = labels[1]
        if y is not None:
            if self.y is not None:
                self.y = torch.cat([self.y, y])
            else:
                self.y = y
        if y_hat is not None:
            if self.y_hat is not None:
                self.y_hat = torch.cat([self.y_hat, y_hat])
            else:
                self.y_hat = y_hat

    def state_dict(self):
        return {
            "y": self.y,
            "y_hat": self.y_hat,
        }

    def load_state_dict(self, state_dict):
        self.y = state_dict["y"]
        self.y_hat = state_dict["y_hat"]
