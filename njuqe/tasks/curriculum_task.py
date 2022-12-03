# Copyright (c) 2020 the NJUQE authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import math
import torch
import numpy as np
from fairseq.data import RawLabelDataset, StripTokenDataset, data_utils, IndexedRawTextDataset
from fairseq import utils
from fairseq.tasks import LegacyFairseqTask, register_task
from njuqe.tasks.qe_task import QEBaseTask
from njuqe.data import CurriculumDataset, RemoveTokenDataset
from fairseq.data import Dictionary, FairseqDataset, data_utils, encoders, iterators
import collections

logger = logging.getLogger(__name__)


class LineRawDataset(IndexedRawTextDataset):
    def __init__(self, tokens, append_eos=False, reverse_order=False):
        self.tokens_list = tokens
        self.lines = []
        self.sizes = []
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        for line in tokens:
            self.sizes.append(len(line))
        self.sizes = np.array(self.sizes)
        self.size = len(self.tokens_list)


def load_curriculum_dataset(
        data_path,
        split,
        src,
        src_dict,
        mt,
        mt_dict,
        generate_epoch,
        mt_tag,
        src_tag,
        tag_dict,
        score,
        align,
        align_args,
        diff,
        bounds,
        combine,
        dataset_impl,
        max_source_positions=512,
        max_target_positions=512,
        left_pad=False,
        shuffle=True,
        append_eos=False,
        prepend_bos=False,
        prepend_score=False,
        joint=False,
):
    prefix = os.path.join(data_path, "{}.{}-{}.".format(split, src, mt))  # train.src-mt

    align_path = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, mt,
                                                              align)) if align is not None else None  # 这里应该是原始文件 不是bin idx那种格式 得手动调文件名 train.src-mt.alignments

    src_dataset = data_utils.load_indexed_dataset(
        prefix + src, src_dict, dataset_impl, combine=combine,
    )
    src_dataset = StripTokenDataset(src_dataset, src_dict.eos())
    # if truncate_source:
    #     src_dataset = TruncateDataset(src_dataset, max_source_positions - 1)

    mt_dataset = data_utils.load_indexed_dataset(
        prefix + mt, mt_dict, dataset_impl, combine=combine,
    )
    # 生成的数据中可能有多个eos，只删除最后一个
    mt_dataset = RemoveTokenDataset(mt_dataset, mt_dict.eos())
    # if truncate_target:
    #     mt_dataset = TruncateDataset(mt_dataset, max_target_positions - 1)

    bounds_path = None
    if bounds == "mt":
        bounds_path = os.path.join(data_path, "{}.{}.bounds".format(split, mt))
    elif bounds == "src":
        bounds_path = os.path.join(data_path, "{}.{}.bounds".format(split, src))
    elif bounds == "both":
        pass  # TODO 加载要加载两种，以后再说

    if mt_tag is not None:
        mt_tag_path = os.path.join(data_path, "{}.{}-None.{}".format(split, mt_tag, mt_tag))  # train.tag-None.tag
        mt_tag_dataset = data_utils.load_indexed_dataset(
            mt_tag_path, tag_dict, dataset_impl, combine=combine,
        )
        mt_tag_dataset = StripTokenDataset(mt_tag_dataset, tag_dict.eos())
        # if truncate_target:
        #     mt_tag_dataset = TruncateDataset(mt_tag_dataset, max_target_positions - 1)
    else:
        mt_tag_dataset = None

    if src_tag is not None:
        src_tag_path = os.path.join(data_path,
                                    "{}.{}-None.{}".format(split, src_tag, src_tag))  # train.source_tag-None.source_tag
        src_tag_dataset = data_utils.load_indexed_dataset(
            src_tag_path, tag_dict, dataset_impl, combine=combine,
        )
        if src_tag_dataset is not None:
            src_tag_dataset = StripTokenDataset(src_tag_dataset, tag_dict.eos())
            # if truncate_source:
            #     src_tag_dataset = TruncateDataset(src_tag_dataset, max_source_positions - 1)
    else:
        src_tag_dataset = None

    def parse_regression_target(line):
        values = line.strip().split()
        return [float(x) for x in values]

    if score is not None:
        score_path = os.path.join(data_path, "{}.{}".format(split, score))  # 这里应该是原始文件 不是bin idx那种格式 得手动调文件名 train.hter
        with open(score_path) as f:
            score_dataset = RawLabelDataset(
                [
                    parse_regression_target(line.strip())
                    for line in f.readlines()
                ]
            )
    else:
        score_dataset = None

    if diff is not None:
        diff_path = os.path.join(data_path, "{}.{}".format(split, diff))  # 这里应该是原始文件 不是bin idx那种格式 得手动调文件名 train.hter
        with open(diff_path) as f:
            diff_dataset = RawLabelDataset(
                np.array([
                    parse_regression_target(line.strip())[0]
                    for line in f.readlines()
                ])
            )
    else:
        diff_dataset = None

    def parse_regression_align(line):
        return [tuple(map(int, x.split('-'))) for x in line.strip().split()]

    # nb_alignments = None
    if align is not None:
        with open(align_path) as f:
            align_dataset = RawLabelDataset(
                [
                    parse_regression_align(line.strip())
                    for line in f.readlines()
                ]
            )
    else:
        align_dataset = None

    def parse_regression_bounds(line):
        values = line.strip().split()
        return [int(x) for x in values]

    if bounds is not None:
        with open(bounds_path) as f:
            bounds_dataset = RawLabelDataset(
                [
                    parse_regression_bounds(line.strip())
                    for line in f.readlines()
                ]
            )
    else:
        bounds_dataset = None

    return CurriculumDataset(
        src_dataset,
        src_dict,
        mt_dataset,
        mt_dict,
        generate_epoch,
        mt_tag_dataset,
        src_tag_dataset,
        tag_dict,
        score_dataset,
        align_dataset,
        align_args,
        diff_dataset=diff_dataset,
        bounds_dataset=bounds_dataset,
        left_pad=left_pad,
        shuffle=shuffle,
        append_eos=append_eos,
        prepend_bos=prepend_bos,
        prepend_score=prepend_score,
        joint=joint,
    )


# TODO 应该拆分成两个函数：获取困难，根据困难过滤。
def filter_indices_by_diff(dataset, indices, competence):
    diff = -dataset.diff_dataset[indices]  # 获取困难度量
    sort_indices = np.argsort(diff, kind="mergesort")
    idx = int(competence * len(indices))
    sort_indices = sort_indices[:idx]
    sort_indices = sort_indices[np.argsort(sort_indices, kind="mergesort")]
    indices = indices[sort_indices]
    return indices


def filter_indices_by_length(dataset, indices, competence):
    idx = int(competence * len(indices))  # indices是按src长度排序从小到大的,competence为0-1，按比例使用
    indices = indices[:idx]  # 返回前面长度小的indices
    return indices


def filter_indices_by_rarity(dataset, indices, competence):
    src_dict = dataset.src_dict
    word_size = sum(src_dict.count)  # 统计src词典中所有的词频总和
    word_rarity = np.array(list(map(lambda x: -math.log(x / word_size), src_dict.count)))  # 计算src词典中每个词的稀有度
    sentence_rarity = []
    for idx in indices:  # 对每句话，求出d_rarity，文中公式3
        sentence_indices = dataset.src_dataset[idx]
        rarity_val = np.sum(word_rarity[sentence_indices], dtype=np.float64)
        sentence_rarity.append(rarity_val)
    sentence_rarity = np.array(sentence_rarity)
    sort_indices = np.argsort(sentence_rarity)  # 按求出的d_rarity排序，重排序indices
    idx = int(competence * len(indices))
    sort_indices = sort_indices[:idx]
    sort_indices = sort_indices[np.argsort(sort_indices, kind="mergesort")]
    indices = indices[sort_indices]
    return indices

def generate_pseudo_dataset(dataset, indices, model):
    tgt_dict = dataset.mt_dict
    tag_dict = dataset.tag_dict
    mt_noise_data = []
    mt_noise_label = []
    mt_hter_label = []

    def get_masked_data(y):
        lm_ids = torch.full_like(y, tgt_dict.pad())
        y_mask = y.clone()
        for i, word in enumerate(y):
            prob = torch.rand(1)
            if prob < 0.15:
                prob = prob / 0.15
                if prob < 100:
                    y_mask[i] = tgt_dict.index("<mask>")
                lm_ids[i] = y[i]
        return y_mask.unsqueeze(0), lm_ids.unsqueeze(0)

    def get_noise_data(gold, pre, lm_ids, k=10):
        pre = pre.topk(k=k, dim=1)
        pre = pre[1]

        lm_ids = lm_ids.tolist()
        gold = gold.tolist()
        noise_data = []
        noise_label = []

        ones_token = torch.ones(k)

        pad_idx = tag_dict.index("PAD")
        ok_idx = tag_dict.index("OK")
        bad_idx = tag_dict.index("BAD")

        noise_data.append(tgt_dict.index("<score>"))
        noise_label.append(pad_idx)
        bad_count = 0.0 + 1e-12
        all_count = 0.0 + 1e-12
        for each_sen_1st, each_sen_gold, each_sen_lm in zip(pre, gold, lm_ids):
            all_count += 1
            choose_one = each_sen_1st[ones_token.multinomial(num_samples=1, replacement=True)]
            if each_sen_lm != tgt_dict.pad() and choose_one != each_sen_lm:
                noise_data.append(choose_one)
                bad_count += 1
                noise_label.append(bad_idx)
            else:
                noise_data.append(each_sen_gold)
                noise_label.append(ok_idx)
        hter_label = bad_count / all_count

        return noise_data, noise_label, hter_label

    for idx in indices:
        src_item = dataset.src_dataset[idx]
        mt_item = dataset.mt_dataset[idx]
        with torch.no_grad():
            mt_tokens_mask, mt_tokens_lmids = get_masked_data(mt_item)  # 首先获取masked-data
            mt_output = model.generator(src_item.unsqueeze(0), None, mt_tokens_mask)[
                0]  # 得到mask词的logits，下一步生成pseudo-data
        noise_data, noise_label, hter_label = get_noise_data(mt_item, mt_output.squeeze(0), mt_tokens_lmids.squeeze(0))
        mt_noise_data.append(torch.IntTensor(noise_data))
        mt_noise_label.append(torch.IntTensor(noise_label))
        mt_hter_label.append(hter_label)

    mt_dataset = LineRawDataset(mt_noise_data)
    mt_dataset = StripTokenDataset(mt_dataset, tgt_dict.eos())
    setattr(dataset, "mt_dataset", mt_dataset)

    mt_tag_dataset = LineRawDataset(mt_noise_label)
    mt_tag_dataset = StripTokenDataset(mt_tag_dataset, tgt_dict.eos())
    setattr(dataset, "mt_tag_dataset", mt_tag_dataset)

    score_dataset = RawLabelDataset(
        [
            [item] for item in mt_hter_label
        ]
    )
    setattr(dataset, "score_dataset", score_dataset)
    pass


@register_task("curriculum")
class QECurriculumTask(QEBaseTask):

    @staticmethod
    def add_args(parser):
        QEBaseTask.add_args(parser)
        parser.add_argument('--by-length', action='store_true', default=False,
                            help='data difficulty by length')
        parser.add_argument('--by-rarity', action='store_true', default=False,
                            help='data difficulty by rarity')
        parser.add_argument('--root', action='store_true', default=False,
                            help='model competence by root')
        parser.add_argument('--max-cl-epoch', default=50, type=int, metavar='N',
                            help='set max curriculum learning epoches')
        # difficulty文件后缀使用diff
        parser.add_argument('--difficulty', default=None, metavar='DIFF',
                            help='difficulty file name symbol for curriculum learning')

    def load_dataset(self, split, combine=False, **kwargs):

        # file name symbol
        src = self.args.src
        mt = self.args.mt
        mt_tag = self.args.mt_tag
        score = self.args.score
        src_tag = self.args.src_tag
        align = self.args.align
        bounds = self.args.bounds
        if split == "train":
            diff = self.args.difficulty # task类中没有该变量
        else:
            diff = None

        data_path = utils.split_paths(self.args.data)[0]

        align_args = None

        if align is not None:
            # 如果有alignments文件的话，那么就初始化align_args，如果没有align，即使window_size=3，那么实际上也是None
            align_args = {"window_size": self.args.window_size, "source_padding_idx": self.src_dict.pad(),
                          "target_padding_idx": self.tgt_dict.pad(), "max_aligned": self.args.max_aligned,
                          "source_unaligned_idx": self.src_dict.index("<unaligned>"),
                          "target_unaligned_idx": self.tgt_dict.index("<unaligned>"),
                          "predict_target": self.args.predict_target, "predict_source": self.args.predict_source,
                          "predict_gaps": self.args.predict_gaps}

        if self.args.no_shuffle:
            shuffle = False
        else:
            shuffle = (split != "test")

        self.datasets[split] = load_curriculum_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            mt,
            self.tgt_dict,
            self.args.generate_epoch,
            mt_tag,
            src_tag,
            self.tag_dict,
            score=score,
            align=align,
            align_args=align_args,
            diff=diff,
            bounds=bounds,
            combine=combine,
            dataset_impl=self.args.dataset_impl,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            left_pad=self.args.left_pad,
            shuffle=shuffle,
            append_eos=self.args.append_eos,
            prepend_bos=self.args.prepend_bos,
            prepend_score=self.args.prepend_score,
            joint=self.args.joint,
        )

    def get_batch_iterator(
            self,
            dataset,
            max_tokens=None,
            max_sentences=None,
            max_positions=None,
            ignore_invalid_inputs=False,
            required_batch_size_multiple=1,
            seed=1,
            num_shards=1,
            shard_id=0,
            num_workers=0,
            epoch=0,
            data_buffer_size=0,
            disable_iterator_cache=False,
    ):
        # 该函数参考自fairseq_task，进行简单重写
        assert isinstance(dataset, FairseqDataset)
        dataset.set_epoch(epoch)

        # 调用使用的dataset的重排序
        is_training = not(epoch==0) # 若epoch=0，表示valid
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices(is_training)

        # 根据设置的最长长度筛选
        if max_positions is not None:
            indices = self.filter_indices_by_size(
                indices, dataset, max_positions, ignore_invalid_inputs
            )

        if is_training:  # valid的epoch默认为0，而train从1开始。
            # 根据论文，此部分句子难度根据长度，模型竞争力使用线性
            linear_initial = 0.05  # 论文中的c0
            linear_competence = (1 - linear_initial) / self.args.max_cl_epoch * epoch + linear_initial  # 论文中公式5，这个if可以改成按args选择
            if self.args.root:
                linear_competence = math.sqrt(
                    (1 - linear_initial ** 2) / self.args.max_cl_epoch * epoch + linear_initial ** 2  # 论文中的公式7
                )
            competence = min(1, linear_competence)  # 当前模型competence

            if self.args.by_length:
                indices = filter_indices_by_length(dataset, indices, competence)  # 按长度，这个if可以改成args判断
            if self.args.by_rarity:
                indices = filter_indices_by_rarity(dataset, indices, competence)  # 按词稀有度
            if self.args.difficulty:
                indices = filter_indices_by_diff(dataset, indices, competence)

        # 使用前面过滤的indices构成batch_sampler
        batch_sampler = dataset.batch_by_size(
            indices,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )

        # 返回可迭代的epoch
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size,
        )

        return epoch_iter

    def begin_epoch(self, epoch, model):
        """Hook function called before the start of each epoch."""
        pass
