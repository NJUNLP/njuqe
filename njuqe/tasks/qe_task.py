# Copyright (c) 2020 the NJUQE authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import torch
import numpy as np
from fairseq import utils
from fairseq.data import (
    Dictionary,
    data_utils,
)
from fairseq.tasks import LegacyFairseqTask, register_task
# from ..data.qe_data_utils import load_qe_dataset
from fairseq.data import (
    RawLabelDataset,
    StripTokenDataset,
    PrependTokenDataset,
    ConcatSentencesDataset,
    NumelDataset,
    NumSamplesDataset,
    TruncateDataset,
    data_utils,
)
from ..data import (
    QEDataset,
)

logger = logging.getLogger(__name__)


def load_qe_dataset(
        data_path,
        split,
        src,
        src_dict,
        mt,
        mt_dict,
        mt_tag,
        src_tag,
        tag_dict,
        score,
        align,
        align_args,
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
    mt_dataset = StripTokenDataset(mt_dataset, mt_dict.eos())
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

    return QEDataset(
        src_dataset,
        src_dict,
        mt_dataset,
        mt_dict,
        mt_tag_dataset,
        src_tag_dataset,
        tag_dict,
        score_dataset,
        align_dataset,
        align_args,
        bounds_dataset=bounds_dataset,
        left_pad=left_pad,
        shuffle=shuffle,
        append_eos=append_eos,
        prepend_bos=prepend_bos,
        prepend_score=prepend_score,
        joint=joint,
    )


class SimpleDict:
    def __init__(self, *args):
        self.pad_index = 2
        self.OK_index = 0
        self.BAD_index = 1

    def pad(self):
        return self.pad_index

    def eos(self):
        return -1


@register_task("qe")
class QEBaseTask(LegacyFairseqTask):
    """
    Reproduce directqe finetune process.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""

        parser.add_argument("data", metavar="FILE", help="file prefix for data")
        parser.add_argument('--src', metavar='SRC',
                            help='source language (source file name symbol)')
        parser.add_argument('--mt', metavar='MT',
                            help='mt language (machine translation file name symbol)')
        parser.add_argument('--mt-tag', default=None, metavar='MT-TAG',
                            help='machine translation tag file name symbol')
        parser.add_argument('--src-tag', default=None, metavar='SRC-TAG',
                            help='source tag file name symbol')
        parser.add_argument('--score', default=None, metavar='SCORE',
                            help='score file name symbol')
        parser.add_argument('--align', default=None, metavar='ALIGN',
                            help='fast_align alignments file name symbol')
        parser.add_argument('--mt-tag-generate', default=None, metavar='MT-TAG-GENERATE',
                            help='mt tag file name symbol when generating fake data')
        parser.add_argument('--mask-symbol', action='store_true', default=False,  # 判断dict中加不加mask，默认不加
                            help='add mask symbol to src and tgt dictionary')
        parser.add_argument('--score-symbol', action='store_true', default=False,
                            help='add score symbol to src and tgt dictionary')
        parser.add_argument('--tag-dict-only', action='store_true', default=False,
                            # 仅加入tag-dict，但是不加入tag data，前面只有tag data时才会加载dict
                            help='load tag dict only')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--left-pad', action='store_true', default=False,
                            help='pad on the left rather than right')
        parser.add_argument('--prepend-bos', action='store_true', default=False,
                            help='prepend bos for src, mt, prepend tag_pad for tag')
        parser.add_argument('--append-eos', action='store_true', default=False,
                            help='append eos for src, mt, append tag_pad for tag')
        parser.add_argument('--prepend-score', action='store_true', default=False,
                            help='prepend score symbol to calculate hter')
        # 这几个参数是针对align文件的
        parser.add_argument('--window_size', default=3, type=int,
                            help='window_size default=3')
        parser.add_argument('--max_aligned', default=5, type=int,
                            help='max_aligned default=5')
        parser.add_argument('--predict-target', action='store_true', default=False,
                            help='make model predict the tag of target token')
        parser.add_argument('--predict-gaps', action='store_true', default=False,
                            help='make model predict the tag of gaps')
        parser.add_argument('--predict-source', action='store_true', default=False,
                            help='make model predict the tag of source token')
        parser.add_argument('--predict-score', action='store_true', default=False,
                            help='make model predict the hter score of sentences')
        parser.add_argument("--qe-destdir", metavar="QEDIR", default="data-qe",
                            help="QE destination dir")
        parser.add_argument("--generate-split", default="train",
                            help="split for QE data generation")
        parser.add_argument("--generate-epoch", default=1, type=int,
                            help="number of epoch for QE data generation")
        parser.add_argument('--qe-meter', action='store_true', default=False,
                            help='calculate qe meter')
        parser.add_argument('--bounds', metavar='BOUNDS', default=None,
                            help='values can be mt / src / both')
        parser.add_argument('--sent-pooling', default="mixed",
                            help='default = last layer, mixed = last layer average + last layer')
        parser.add_argument('--joint', action='store_true', default=False,
                            help='joint tokens for encoder models, like xlmr')
        parser.add_argument('--no-shuffle', action='store_true', default=False,
                            help='whether shuffle')

    def __init__(self, args, src_dict, tgt_dict=None, tag_dict=None):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.tag_dict = tag_dict
        self.model = None

    @classmethod
    def load_dictionary(cls, filename, align=False, mask=False, tag=False, score=False):
        """Load the dictionary from the filename

        Args:
            :param filename: str, the filename of dictionary
            :param align: bool, whether add "<unaligned>"
            :param mask: bool, whether add "<mask>"
            :param tag: bool, tag_dict have "OK", "BAD", and "PAD" beside nspecial
        """

        dictionary = Dictionary.load(filename)
        # tag_dict have OK, BAD, and PAD beside nspecial
        if tag:
            dictionary.add_symbol("PAD")
        if align is not None:
            dictionary.add_symbol("<unaligned>")
        if mask:
            dictionary.add_symbol("<mask>")
        if score:
            dictionary.add_symbol("<score>")

        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        path = utils.split_paths(args.data)[0]

        align = args.align
        mask = args.mask_symbol
        score = args.score_symbol

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(path, "dict.{}.txt".format(args.src)),
            align,
            mask,
            tag=False,
            score=score,
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(path, "dict.{}.txt".format(args.mt)),
            align,
            mask,
            tag=False,
            score=score,
        )
        tag_dict = None
        if args.mt_tag is not None:
            tag_dict = cls.load_dictionary(
                os.path.join(path, "dict.{}.txt".format(args.mt_tag)),
                tag=True,
            )
        elif args.tag_dict_only:  # 如果仅仅加载字典，那么也load tag-dict,固定命名为dict.tags.txt了，但是下面loggerinfo最一后一行比较丑
            tag_dict = cls.load_dictionary(
                os.path.join(path, "dict.tags.txt"),
                tag=True,
            )

        logger.info("[{}] dictionary: {} types".format(args.src, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(args.mt, len(tgt_dict)))
        if tag_dict is not None:
            logger.info("[{}] dictionary: {} types".format(args.mt_tag, len(tag_dict)))

        return cls(args, src_dict, tgt_dict, tag_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        # file name symbol
        src = self.args.src
        mt = self.args.mt
        mt_tag = self.args.mt_tag
        score = self.args.score
        src_tag = self.args.src_tag
        align = self.args.align
        bounds = self.args.bounds

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

        self.datasets[split] = load_qe_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            mt,
            self.tgt_dict,
            mt_tag,
            src_tag,
            self.tag_dict,
            score=score,
            align=align,
            align_args=align_args,
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

        return self.datasets[split]

    def build_model(self, args):  #
        model = super().build_model(args)

        if self.args.arch == 'xlmqe_base':
            model.register_classification_head(
                getattr(self.args, "classification_head_name", "xlm_qe_head"),
                num_classes=self.args.num_classes,
            )
        self.model = model
        return model

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return self.args.max_source_positions, self.args.max_target_positions

    def generate_step(self, sample, model):
        model.train()
        with torch.no_grad():
            outputs = model(**sample["net_input"], generate=True)
        return outputs

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    @property
    def tag_dictionary(self):
        """Return the tag :class:`~fairseq.data.Dictionary`."""
        return self.tag_dict
