# Copyright (c) 2020 the NJUQE authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import numpy as np
import torch
from fairseq.data import (
    FairseqDataset,
    IdDataset,
    data_utils,
)

logger = logging.getLogger(__name__)


def collate(
        samples,
        pad_token,
        tag_pad_token,
        left_pad=False,
        pad_to_length=None,
        align_args=None,
        shuffle=True,
        joint=False,
):
    if len(samples) == 0:
        return {}

    def merge(key, pad_token, left_pad=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_token,
            left_pad,
            pad_to_length=pad_to_length,
        )

    def merge_align(key, pad_token, left_pad=False):
        return collate_alignment_tokens(
            [s[key] for s in samples],
            pad_token,
            left_pad,
        )

    src = merge(
        "src",
        pad_token=pad_token,
        left_pad=left_pad,
        pad_to_length=pad_to_length["src"] if pad_to_length is not None else None,
    )
    # sort by descending source length
    src_lengths = torch.LongTensor(
        [s["src"].ne(pad_token).long().sum() for s in samples]
    )
    src_lengths, sort_order = src_lengths.sort(descending=True)
    if shuffle:
        src_lengths, sort_order = src_lengths.sort(descending=True)     # 其实这个collater中不需要sort-order这一步了，前面都有order_indices?
    else:
        sort_order = torch.arange(0, len(src_lengths))
    src = src.index_select(0, sort_order)
    # 如果想用ntokens做归一，应该不同tag分别统计
    ntokens = src_lengths.sum().item()

    id = torch.LongTensor([s["id"] for s in samples])
    id = id.index_select(0, sort_order)

    mt = merge(
        "mt",
        pad_token=pad_token,
        left_pad=left_pad,
        pad_to_length=pad_to_length["mt"] if pad_to_length is not None else None,
    )
    mt = mt.index_select(0, sort_order)

    # 初始化其他可能读入的文件
    mt_tag = None  # mt的tag文件
    mt_gap_tag = None  # mt_gap的tag文件
    src_tag = None  # src的tag文件
    score = None  # score文件
    joint_tokens = None  # mt+src的拼接文件
    bounds = None  # 目前是mt的tag对应关系
    masked_tokens = None  # 目前是mt+src对应的pad关系

    if samples[0].get("mt_tag", None) is not None:
        mt_tag = merge(
            "mt_tag",
            pad_token=tag_pad_token,
            left_pad=left_pad,
            pad_to_length=pad_to_length["mt_tag"] if pad_to_length is not None else None,
        )
        mt_tag = mt_tag.index_select(0, sort_order)

    if samples[0].get("mt_gap_tag", None) is not None:
        mt_gap_tag = merge(
            "mt_gap_tag",
            pad_token=tag_pad_token,
            left_pad=left_pad,
            pad_to_length=pad_to_length["mt_gap_tag"] if pad_to_length is not None else None,
        )
        mt_gap_tag = mt_gap_tag.index_select(0, sort_order)

    if samples[0].get("src_tag", None) is not None:
        src_tag = merge(
            "src_tag",
            pad_token=tag_pad_token,
            left_pad=left_pad,
            pad_to_length=pad_to_length["src_tag"] if pad_to_length is not None else None,
        )
        src_tag = src_tag.index_select(0, sort_order)

    if samples[0].get("score", None) is not None:
        score = torch.FloatTensor([s["score"] for s in samples])
        score = score.index_select(0, sort_order)

    if samples[0].get("bounds", None) is not None:
        bounds = merge(
            "bounds",
            pad_token=-1,  # 没有对应的那么用-1表示
            left_pad=left_pad,
            pad_to_length=pad_to_length["bounds"] if pad_to_length is not None else None,
        )
        bounds = bounds.index_select(0, sort_order)

    if joint:
        if src is not None and mt is not None:
            joint_tokens = torch.cat((mt, src), 1)
            masked_tokens = ~ joint_tokens.eq(pad_token)

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "mt_tag": mt_tag,
        "mt_gap_tag": mt_gap_tag,
        "src_tag": src_tag,
        "score": score,
        "net_input": {
            "src_tokens": src,
            "mt_tokens": mt,
            "joint_tokens": joint_tokens,
            "bounds": bounds,
            "masked_tokens_sing": masked_tokens,
        },
    }

    # TODO：修改丑陋的align_args
    if samples[0].get("align", None) is not None:
        align = merge_align(
            "align",
            pad_token=pad_token,
            left_pad=left_pad,
        )
        align = align.index_select(0, sort_order)
        target_input, source_input, align_nb = make_input(mt, src, align, align_args)
        batch["net_input"]["src_tokens"] = source_input
        batch["net_input"]["mt_tokens"] = target_input
        batch["net_input"]["align_nb"] = align_nb
    return batch


class QEDataset(FairseqDataset):
    """
    A wrapper around torch.utils.data.Dataset for QE data.
    Args:
        src_dataset (torch.utils.data.Dataset): source dataset to wrap
        src_dict (~fairseq.data.Dictionary): source vocabulary
        mt_dataset (torch.utils.data.Dataset, optional): target dataset to wrap
        mt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        mt_tag_dataset (torch.utils.data.Dataset, optional): tag mt dataset to wrap
        src_tag_dataset (torch.utils.data.Dataset, optional): tag src dataset to wrap
        tag_dict (~fairseq.data.Dictionary, optional): tag vocabulary
        score_dataset (list of list of score, optional): score dataset to wrap
        left_pad (bool, optional): pad tensors on the left side rather than right
            (default: False).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
    """

    def __init__(
            self,
            src_dataset,
            src_dict,
            mt_dataset,
            mt_dict,
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
        super(QEDataset, self).__init__()
        self.src_dataset = src_dataset
        self.src_dict = src_dict
        self.mt_dataset = mt_dataset
        self.mt_dict = mt_dict
        self.mt_tag_dataset = mt_tag_dataset
        self.src_tag_dataset = src_tag_dataset
        self.tag_dict = tag_dict
        self.score_dataset = score_dataset
        self.align_dataset = align_dataset
        self.align_args = align_args
        self.diff_dataset = diff_dataset
        self.left_pad = left_pad
        # the pad could be different of src and mt in few cases, but language_pair_dataset do like this.
        self.pad_token = self.src_dict.pad()
        self.score_token = self.mt_dict.index("<score>")
        # tag_dict have "OK", "BAD", and "PAD" beside nspecial
        self.tag_pad_token = self.tag_dict.index("PAD") - self.tag_dict.nspecial if self.tag_dict is not None else self.pad_token
        self.shuffle = shuffle
        self.append_eos = append_eos
        self.prepend_bos = prepend_bos
        self.prepend_score = prepend_score
        self.bounds_dataset = bounds_dataset
        self.joint = joint

    def get_batch_shapes(self):
        pass

    def __getitem__(self, index):

        def prepend_token(item, token):
            res = None
            if item is not None:
                res = torch.cat([torch.LongTensor([token]), item])
            return res

        def append_token(item, token):
            res = None
            if item is not None:
                res = torch.cat([item, torch.LongTensor([token])])
            return res

        src_item = self.src_dataset[index]
        mt_item = self.mt_dataset[index] if self.mt_dataset is not None else None
        align_item = self.align_dataset[index] if self.align_dataset is not None else None
        bounds_item = torch.LongTensor(self.bounds_dataset[index]) if self.bounds_dataset is not None else None
        # tag_dict have "OK", "BAD", and "PAD" beside nspecial
        if self.mt_tag_dataset is not None:
            mt_tag_item = self.mt_tag_dataset[index] - self.tag_dict.nspecial
            if bounds_item is not None and len(mt_tag_item) == len(bounds_item):  # 表明是用sentencepiece的，和bounds数目一样，只有tag
                mt_gap_tag_item = None
            elif bounds_item is not None and len(mt_tag_item) == len(bounds_item) + 1:
                mt_gap_tag_item = mt_tag_item
                mt_tag_item = None
            elif bounds_item is not None and len(mt_tag_item) == len(bounds_item) * 2 + 1:
                mt_tag_all = mt_tag_item
                mt_gap_tag_item = mt_tag_all[::2]
                mt_tag_item = mt_tag_all[1::2]
            elif len(mt_tag_item) == len(mt_item):  # 表示输入的只有mt的tag
                mt_gap_tag_item = None
            elif len(mt_tag_item) == len(mt_item) + 1:  # 表示输入的只有gap的tag
                mt_gap_tag_item = mt_tag_item
                mt_tag_item = None
            elif len(mt_tag_item) == len(mt_item) * 2 + 1:  # mt和gap都有
                mt_tag_all = mt_tag_item
                mt_gap_tag_item = mt_tag_all[::2]
                mt_tag_item = mt_tag_all[1::2]
            else:
                raise Exception(
                    "The length of tag must be l(mt tag), l+1(mt gap tag), "
                    "2l+1(interlaced mt and mt gap), where l is the length of mt."
                )
        else:
            mt_tag_item = None
            mt_gap_tag_item = None

        src_tag_item = self.src_tag_dataset[
                           index] - self.tag_dict.nspecial if self.src_tag_dataset is not None else None
        if self.score_dataset is not None:
            score_item = self.score_dataset[index]
        else:
            score_item = None

        if self.prepend_score:
            mt_item = prepend_token(mt_item, self.score_token)
            mt_tag_item = prepend_token(mt_tag_item, self.tag_pad_token)
            bounds_item = torch.cat(
                (torch.zeros(1, dtype=torch.long), bounds_item + 1)
            ) if bounds_item is not None else None

        if self.append_eos:
            src_item = append_token(src_item, self.src_dict.eos())
            mt_item = append_token(mt_item, self.mt_dict.eos())
            # if align_item is not None, append_eos can not be true
            mt_tag_item = append_token(mt_tag_item, self.tag_pad_token)
            # mt_gap_tag did not need to be add extra pad
            src_tag_item = append_token(src_tag_item, self.tag_pad_token)
            bounds_item = torch.cat(
                (bounds_item, torch.tensor([len(mt_item) - 1], dtype=torch.long))  # TODO，不一定是mt，也可能是src的tags
            ) if bounds_item is not None else None

        if self.prepend_bos:
            src_item = prepend_token(src_item, self.src_dict.bos())
            mt_item = prepend_token(mt_item, self.mt_dict.bos())
            # if align_item is not None, prepend_bos can not be true
            mt_tag_item = prepend_token(mt_tag_item, self.tag_pad_token)
            # mt_gap_tag did not need to be add extra pad
            src_tag_item = prepend_token(src_tag_item, self.tag_pad_token)
            bounds_item = torch.cat((
                torch.zeros(1, dtype=torch.long), bounds_item + 1)
            ) if bounds_item is not None else None

        example = {
            "id": index,
            "src": src_item,
            "mt": mt_item,
            "align": align_item,
            "mt_tag": mt_tag_item,
            "mt_gap_tag": mt_gap_tag_item,
            "src_tag": src_tag_item,
            "score": score_item,
            "bounds": bounds_item,
        }

        return example

    def __len__(self):
        return len(self.src_dataset)

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'src': source_pad_to_length, 'mt': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.
        Returns:
            dict: a mini-batch with the following keys:
                - `id` (LongTensor): example IDs in the original input order
                - `nsentences` (int): total number of sentences in the batch
                - `ntokens` (int): total number of tokens in the batch
                - `src` (LongTensor): a padded 2D Tensor of tokens in the source
                    sentence of shape `(bsz, src_len)`.
                - `mt` (LongTensor): a padded 2D Tensor of tokens in the target
                    sentence of shape `(bsz, mt_len)`.
                - `mt_tag` (LongTensor): a padded 2D Tensor of tokens in the tag
                    sentence of shape `(bsz, tag_len) or (bsz, tag_len*2+1)`.
                - `src_tag` (LongTensor): a padded 2D Tensor of tokens in the tag
                    sentence of shape `(bsz, src_len)`.
                - `score` (FloatTensor): a 2D Tensor of scores of shape
                    `(bsz, score_len)`.
        """
        res = collate(
            samples,
            self.pad_token,
            self.tag_pad_token,
            left_pad=self.left_pad,
            pad_to_length=pad_to_length,
            align_args=self.align_args,
            shuffle=self.shuffle,
            joint=self.joint,
        )
        return res

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(
            self.src_dataset.sizes[index],
            self.mt_dataset.sizes[index] if self.mt_dataset.sizes is not None else 0,
        )

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.src_dataset.sizes[index],
            self.mt_dataset.sizes[index] if self.mt_dataset.sizes is not None else 0,
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
            # sort by target length, then source length
            if self.mt_dataset.sizes is not None:
                indices = indices[np.argsort(self.mt_dataset.sizes[indices], kind="mergesort")]
            return indices[np.argsort(self.src_dataset.sizes[indices], kind="mergesort")]
        else:
            indices = np.arange(len(self), dtype=np.int64)
            return indices  # 微调了一下，因为在test时不用shuffle的

    @property
    def supports_prefetch(self):
        pass

    def prefetch(self, indices):
        pass

    def filter_indices_by_size(self, indices, max_sizes):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.
        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and mt (then list or tuple)
        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        return data_utils.filter_paired_dataset_indices_by_size(
            self.src_dataset.sizes,
            self.mt_dataset.sizes,
            indices,
            max_sizes,
        )
