# Copyright (c) 2020 the NJUQE authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from njuqe.data import QEDataset
import logging

logger = logging.getLogger(__name__)


class CurriculumDataset(QEDataset):

    def __init__(
            self,
            src_dataset,
            src_dict,
            mt_dataset,
            mt_dict,
            generate_epoch,
            mt_tag_dataset=None,
            src_tag_dataset=None,
            tag_dict=None,
            score_dataset=None,
            align_dataset=None,
            align_args=None,
            diff_dataset=None,
            bounds_dataset=None,
            left_pad=False,
            shuffle=True,
            append_eos=False,
            prepend_bos=False,
            prepend_score=False,
            joint=False,
    ):
        QEDataset.__init__( # 原来的初始化写法不对，会重复初始化变量
            self,
            src_dataset,
            src_dict,
            mt_dataset,
            mt_dict,
            mt_tag_dataset=mt_tag_dataset,
            src_tag_dataset=src_tag_dataset,
            tag_dict=tag_dict,
            score_dataset=score_dataset,
            align_dataset=align_dataset,
            align_args=align_args,
            diff_dataset=diff_dataset,
            bounds_dataset=bounds_dataset,
            left_pad=left_pad,
            shuffle=shuffle,
            append_eos=append_eos,
            prepend_bos=prepend_bos,
            prepend_score=prepend_score,
            joint=joint,
        )
        self.generate_epoch = generate_epoch

    def ordered_indices(self, is_training):
        # 该部分用于对index排序，是否需要保留或扩展待定
        if self.shuffle and is_training:
            generate_epoch = self.generate_epoch
            assert len(self) % generate_epoch == 0 # 如果是valid的集，只有1epoch，这样就一定会assert
            dataset_size = int(len(self) / generate_epoch) # dataset_size不能为float类型
            indices_list = []
            for epoch in range(generate_epoch):
                tmp_indices = np.random.permutation(dataset_size).astype(np.int64) + int(dataset_size*epoch)
                # sort by target length, then source length
                if self.mt_dataset.sizes is not None:
                    tmp_indices = tmp_indices[np.argsort(self.mt_dataset.sizes[tmp_indices], kind="mergesort")]
                tmp_indices = tmp_indices[np.argsort(self.src_dataset.sizes[tmp_indices], kind="mergesort")]
                indices_list.append(tmp_indices)
            indices = np.concatenate(indices_list,axis=0)
            return indices
        elif self.shuffle and not is_training:
            indices = np.random.permutation(len(self)).astype(np.int64)
            if self.mt_dataset.sizes is not None:
                indices = indices[np.argsort(self.mt_dataset.sizes[indices], kind="mergesort")]
            return indices[np.argsort(self.src_dataset.sizes[indices], kind="mergesort")]
        else:
            indices = np.arange(len(self), dtype=np.int64)
            return indices