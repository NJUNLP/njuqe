# Copyright (c) 2020 the NJUQE authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.data import BaseWrapperDataset


class RemoveTokenDataset(BaseWrapperDataset):
    def __init__(self, dataset, id_to_strip):
        super().__init__(dataset)
        self.id_to_strip = id_to_strip

    def __getitem__(self, index):
        item = self.dataset[index]
        if len(item) > 0 and item[-1] == self.id_to_strip:
            item = item[:-1]
        return item
