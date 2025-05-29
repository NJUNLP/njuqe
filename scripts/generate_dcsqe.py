#!/usr/bin/env python3 -u
# !/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from argparse import Namespace
from collections import defaultdict
from copy import deepcopy
from itertools import chain

import time
import math
import random
from typing import List, Tuple, Set

import stanza
import torch
import numpy as np
from stanza import DownloadMethod
from stanza.models.common.doc import Word

from fairseq import checkpoint_utils, distributed_utils, options, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import metrics, progress_bar
from omegaconf import DictConfig
from fairseq.data import indexed_dataset
from fairseq_cli.preprocess import dataset_dest_file


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("njuqe_cli.data_generation")


def main(cfg: DictConfig, override_args=None):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)
    random.seed(cfg.common.seed)

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    use_fp16 = cfg.common.fp16
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    if use_cuda:
        torch.cuda.set_device(cfg.distributed_training.device_id)

    if override_args is not None:
        overrides = vars(override_args)
        overrides.update(eval(getattr(override_args, "model_overrides", "{}")))
    else:
        overrides = None

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [cfg.common_eval.path],
        arg_overrides=overrides,
        suffix=cfg.checkpoint.checkpoint_suffix,
    )
    model = models[0]

    # Move models to GPU
    for model in models:
        if use_fp16:
            model.half()
        if use_cuda:
            # pass
            model.cuda()

    # Print args
    logger.info(saved_cfg)

    try:
        subset = cfg.task.gen_subset
        task.load_dataset(subset, combine=False, epoch=1, task_cfg=saved_cfg.task)
        dataset = task.dataset(subset)
    except KeyError:
        raise Exception("Cannot find dataset: " + subset)

    prefix = "{}{}".format(subset, cfg.task.seed)

    # results_path = "{}/seed{}".format(cfg.task.results_path, cfg.task.seed)
    # os.makedirs(results_path, exist_ok=True)
    fake_hyp_file = "{}/{}.hyp".format(cfg.task.results_path, prefix)
    fake_hyp_file = open(fake_hyp_file, "w")
    fake_skip_file = "{}/{}.skip".format(cfg.task.results_path, prefix)
    fake_skip_file = open(fake_skip_file, "w")
    # fake_dtag_file = "{}/{}.dag".format(cfg.task.results_path, prefix)
    # fake_dtag_file = open(fake_dtag_file, "w")
    # fake_mqm_file = "{}/{}.mqm_weight".format(cfg.task.results_path, prefix)
    # fake_mqm_file = open(fake_mqm_file, "w")

    error_weight_dict = {1: 0.1, 2: 1, 3: 5}
    error_name_dict = {1: "minor", 2: "major", 3: "critical"}
    skip_numbers = []

    # Initialize data iterator
    itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[m.max_positions() for m in models],
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)

    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        prefix=f"generate QE data on '{subset}' subset",
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    nlp = stanza.Pipeline(lang=cfg.task.target_lang[:2], processors='tokenize,pos,lemma,depparse', verbose=False,
                          download_method=DownloadMethod.REUSE_RESOURCES, tokenize_pretokenized='True',
                          dir="/home/nfs03/laizj/model/stanza_resources", use_gpu=use_cuda)

    for i, sample in enumerate(progress):
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        # extra randomness by drop out
        # model.train()
        model.eval()
        with torch.no_grad():
            target = sample["target"].unsqueeze(-1)
            logits = model(**sample["net_input"], return_all_hiddens=False)[0]
            ok_mask = sample["tag"].ne(task.tag_dict.index("BAD"))

            prob = torch.softmax(logits, dim=-1)
            prob = torch.gather(prob, dim=-1, index=target).squeeze(-1)
            tags = torch.zeros_like(sample["tag"]) + 3
            # tags[prob >= 0.004] = 2
            # tags[prob >= 0.04] = 1
            # tags[prob >= 0.2] = 0
            # tags[ok_mask] = 0

            _, indices = logits.topk(k=30, dim=-1)
            tags = torch.zeros_like(sample["tag"]) + 3
            ok2_mask = target.eq(indices[:, :, :4]).any(dim=-1)
            tags[ok2_mask] = 0
            minor_mask = target.eq(indices[:, :, 4:8]).any(dim=-1)
            tags[minor_mask] = 1
            major_mask = target.eq(indices[:, :, 8:]).any(dim=-1)
            tags[major_mask] = 2
            tags[ok_mask] = 0

        for j, sample_id in enumerate(sample["id"].tolist()):

            if task.target_dictionary.unk() in sample["target"][j]:
                skip_numbers.append(sample_id)
                continue
            final_words = []
            final_tags = []
            severity = 0
            word = ""
            for token, tag in zip(sample["target"][j], tags[j]):
                # 允许单词遍历到eos，使得可以获得<EOS>的标签
                if token == task.target_dictionary.pad():
                    break
                severity = max(severity, tag.item())
                token = task.target_dictionary[token]
                if token.endswith("@@"):
                    word += token[:-2]
                else:
                    word += token
                    final_words.append(word)
                    final_tags.append(severity)
                    word = ""
                    severity = 0

            assert word == "", "生成的句子一定是完整的，不是以@@分词结尾"

            if final_words[-1] == "</s>":
                final_words = final_words[:-1]

            # 消融span使用
            no_span_tag_str = []
            no_span_dtag_str = []
            no_span_mqm_score = 0
            last = 0
            for tag in final_tags:
                if tag != last and last != 0:
                    no_span_mqm_score += error_weight_dict[last]
                last = tag
                if tag == 0:
                    no_span_tag_str.append("OK")
                    no_span_dtag_str.append("OK")
                else:
                    no_span_tag_str.append("BAD")
                    no_span_dtag_str.append(error_name_dict[tag])
            no_span_dtag_str = no_span_dtag_str[:-1]
            assert len(no_span_dtag_str) == len(final_words)

            src_str = task.source_dictionary.string(sample["net_input"]["src_tokens"][j], extra_symbols_to_ignore=[task.source_dictionary.pad()])
            tgt_str = task.target_dictionary.string(sample["target"][j], extra_symbols_to_ignore=[task.target_dictionary.pad()])

            nodes, num_words = create_tree(nlp, final_words)
            PreprocessAncestors(nodes[0], num_words + 1)

            final_tags = torch.tensor(final_tags, dtype=torch.long)
            start = 0
            error_nums = 1
            mqm_score = 0
            idx = 1
            while idx < num_words:

                if (final_tags[start] != 0) != (final_tags[idx] != 0):
                    if final_tags[start].item() != 0:
                        left, right = findSpanCoverSegment(nodes, {nodes[start + 1], nodes[idx]})
                        assert left <= start, f"生成的[{left}, {right})短语段未囊括[{start}, {idx})"
                        severity = final_tags[left: right].max().item()
                        final_tags[left: right] = severity
                        mqm_score += error_weight_dict[severity]
                        error_nums = error_nums + 1
                        idx = right
                        start = right
                    else:
                        start = idx
                idx += 1

            if start < num_words and final_tags[start].item() != 0:  # Only deal Bad Segment
                left, right = findSpanCoverSegment(nodes, {nodes[start + 1], nodes[idx]})
                assert left <= start, f"生成的[{left}, {right})短语段未囊括[{start}, {idx})"
                severity = final_tags[left: right].max().item()
                final_tags[left: right] = final_tags[left: right].max()
                mqm_score += error_weight_dict[severity]
                error_nums = error_nums + 1

            no_span_mqm_score = 1 - no_span_mqm_score / num_words
            mqm_score = 1 - mqm_score / num_words
            tag_str = []
            dtag_str = []

            for tag in final_tags:
                if tag.item() == 0:
                    tag_str.append("OK")
                    dtag_str.append("OK")
                else:
                    tag_str.append("BAD")
                    dtag_str.append(error_name_dict[tag.item()])
            dtag_str = dtag_str[:-1]
            assert len(dtag_str) == len(final_words)

            fake_hyp_file.write(f'H-{sample_id}\t{mqm_score}\t{" ".join(tag_str) }\t{" ".join(dtag_str)}\t{no_span_mqm_score}\t{" ".join(no_span_tag_str) }\t{" ".join(no_span_dtag_str)}\t{src_str}\t{tgt_str}\n')

        # progress.log({}, step=i)
        logger.info("Generate iter {}".format(i))

    fake_hyp_file.close()
    fake_skip_file.write("\n".join(map(str, skip_numbers)))
    fake_skip_file.close()

    logger.info("Data generation done!")


class Node:
    def __init__(self, index):
        self.index = index          # Unique index of the node
        self.children = []          # List of child nodes
        self.parent = None          # Parent node
        self.depth = 0              # Depth of the node in the tree
        self.ancestors = []         # Ancestor table for binary lifting (LCA computation)

    def add_child(self, child):
        """Adds a child node to the current node."""
        self.children.append(child)
        child.parent = self
        child.depth = self.depth + 1


def create_tree(nlp, final_words):
    """
    Creates a sample tree for testing purposes.
    """
    doc = nlp(" ".join(final_words))
    # print(*[
    #     f'id: {word.id}\tword: {word.text}\thead id: {word.head}\t'
    #     f'head: {sent.words[word.head - 1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}'
    #     for sent in doc.sentences for word in sent.words], sep='\n')

    # Initialize
    sent = doc.sentences[0]
    words = sent.words  # words.id start from index 1
    num_words = len(words)
    nodes = [Node(i) for i in range(num_words + 1)]

    for idx, word in enumerate(sent.words):

        # Add Dependencies edges if required mask whole span and the word isn't the root
        nodes[word.head].add_child(nodes[idx + 1])
    return nodes, num_words


def PreprocessAncestors(root, N):
    """Preprocesses ancestor tables for all nodes to enable efficient LCA queries."""
    MAX_LOG_N = N.bit_length()  # Maximum depth for ancestor tables

    def dfs(node, parent):
        if parent:
            node.depth = parent.depth + 1
        else:
            node.depth = 0
        node.ancestors = [None] * MAX_LOG_N
        node.ancestors[0] = parent
        for k in range(1, MAX_LOG_N):
            if node.ancestors[k - 1]:
                node.ancestors[k] = node.ancestors[k - 1].ancestors[k - 1]
            else:
                node.ancestors[k] = None
        for child in node.children:
            dfs(child, node)

    dfs(root, None)


def LCA(u, v):
    """Computes the Lowest Common Ancestor (LCA) of two nodes u and v."""
    if u.depth < v.depth:
        u, v = v, u
    # Bring u up to v's depth
    for k in reversed(range(len(u.ancestors))):
        if u.ancestors[k] and u.ancestors[k].depth >= v.depth:
            u = u.ancestors[k]
    if u == v:
        return u
    # Find the LCA
    for k in reversed(range(len(u.ancestors))):
        if u.ancestors[k] != v.ancestors[k]:
            u = u.ancestors[k]
            v = v.ancestors[k]
    return u.parent


def getSubtree(node):
    """Returns all descendants of the node, including the node itself."""
    result = set()

    def dfs(n):
        result.add(n)
        for child in n.children:
            dfs(child)

    dfs(node)
    return result


def add_nodes_between(lca_node, node, current_set):
    """Adds all nodes between the LCA and the given node to the current set."""
    while node != lca_node:
        node = node.parent
        current_set.add(node)
    current_set.add(lca_node)


def findSpanCoverSegment(nodes, S):
    """Finds the minimal covering span for a given set of nodes S."""
    current_set = set(S)
    while True:

        # Step 1: Compute the Lowest Common Ancestor (LCA) of all nodes in the current set
        nodes_list = list(current_set)
        lca_node = nodes_list[0]
        for node in nodes_list[1:]:
            lca_node = LCA(lca_node, node)

        # Step 2: Add nodes between LCA and each node in the current set
        previous_set = current_set.copy()
        for node in previous_set:
            add_nodes_between(lca_node, node, current_set)

        # Step 3: Find the minimum and maximum indices in the current set
        indices = [node.index for node in current_set]
        min_index = min(indices)
        max_index = max(indices)

        # Step 4: Add all nodes between min_index and max_index to the current set
        for i in range(min_index, max_index + 1):
            current_set.add(nodes[i])

        # Terminate if the set doesn't change
        if current_set == previous_set:
            break

    current_set = [node.index for node in current_set]
    return min(current_set) - 1, max(current_set)


def cli_main():
    parser = options.get_validation_parser()
    args = options.parse_args_and_arch(parser)

    # only override args that are explicitly given on the command line
    override_parser = options.get_validation_parser()
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)

    distributed_utils.call_main(convert_namespace_to_omegaconf(args), main, override_args=override_args)


if __name__ == "__main__":
    cli_main()
