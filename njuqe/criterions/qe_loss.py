# Copyright (c) 2020 the NJUQE authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import copy
import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
import numpy as np
from njuqe.logging.metrics import log_labels
from .qe_metric import (
    get_predictions_flat,
    make_loss_weights,
    compute_pearsonr,
    get_f1_compute_fn,
)

def label_smoothed_nll_loss(lprobs, target, epsilon, weight=None, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        temp_target = copy.deepcopy(target)
        temp_target.masked_fill_(pad_mask, 0)
        nll_loss = -lprobs.gather(dim=-1, index=temp_target)

        if weight is not None:
            weight = weight.expand_as(lprobs)
            weight = weight.gather(dim=-1, index=temp_target)
            nll_loss = weight * nll_loss
            smooth_loss = (-lprobs * weight).sum(dim=-1, keepdim=True)
        else:
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)

    else:
        nll_loss = -lprobs.gather(dim=-1, index=target)
        if weight is not None:
            weight = weight.expand_as(lprobs)
            weight = weight.gather(dim=-1, index=target)
            nll_loss = weight * nll_loss
            smooth_loss = (-lprobs * weight).sum(dim=-1, keepdim=True)

        else:
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    if reduce:  # 加法reduce
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()

    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

@register_criterion("qe_base")
class QEBaseCriterion(FairseqCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            score_loss_weight,
            ok_loss_weight,
            fine_tune,
    ):
        super().__init__(task)
        self.eps = label_smoothing
        self.score_loss_weight = score_loss_weight
        self.ok_weight = ok_loss_weight
        self.sentence_avg = sentence_avg

        self.tag_pad = task.tgt_dict.index("<pad>")
        self.weight = None
        tag_dict = task.tag_dict
        if tag_dict is not None:
            self.ok_idx = tag_dict.index("OK") - tag_dict.nspecial
            self.bad_idx = tag_dict.index("BAD") - tag_dict.nspecial
            self.tag_pad = tag_dict.index("PAD") - tag_dict.nspecial
            # TODO: 修改为适应n类
            self.weight = make_loss_weights(2, self.ok_idx, self.ok_weight)
        self.fine_tune = fine_tune

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--score-loss-weight', default=1., type=float, metavar='D',
                            help='weight of score loss')
        parser.add_argument('--ok-loss-weight', default=1., type=float, metavar='D',
                            help='weight of ok loss')
        parser.add_argument("--fine-tune", action="store_true", default=False)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # outputs = {
        #             "mt_tag": {
        #                 "out": ,
        #                 "tgt": ,
        #             },
        #             "mt_gap_tag": {
        #                 "out": ,
        #                 "tgt": ,
        #             },
        #             "src_tag": {
        #                 "out": ,
        #                 "tgt": ,
        #             },
        #             "score": {
        #                 "out": ,
        #                 "tgt": ,
        #             },
        #         }
        # tgt for pseudo qe data methods
        # outputs must have at least one output of four label
        if self.fine_tune:
            outputs = model(**sample["net_input"], fine_tune=self.fine_tune)
        else:
            outputs = model(**sample["net_input"])

        losses, nll_losses = [], []

        # TODO:使用不同权重加权
        for obj in outputs:
            net_output = outputs[obj].get("out", None)
            target = sample.get(obj, None)
            sample_size = (
                target.size(0) if self.sentence_avg else sample["ntokens"]
            )
            if target is None:
                target = outputs[obj].get("tgt", None)
            tmp_loss = self.compute_loss(
                obj,
                model,
                net_output,
                target,
                self.weight,
                ignore_index=self.tag_pad,
                reduce=reduce,
            )
            if obj == "score":
                y = target.view(-1)
                y_hat = net_output.view(-1)
            else:
                y, y_hat = get_predictions_flat(target, net_output, self.tag_pad)
            tmp_loss["labels"] = {"y": y, "y_hat": y_hat}
            losses = losses + [tmp_loss]
            nll_losses = nll_losses + [tmp_loss.get("nll_loss", 0.0)]

        loss = sum(l["loss"] for l in losses)
        # tag_nll_loss = sum(l for l in nll_losses) if len(nll_losses) > 0 else loss.new_tensor(0)
        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = target.size(0)
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
        }

        for l in losses:
            logging_output[l["name"] + "_loss"] = (
                utils.item(l["loss"].data / l["factor"])
                if reduce
                else l[["loss"]].data / l["factor"]
            )
            if self.fine_tune and self.task.args.qe_meter:
                logging_output[l["name"] + "_labels"] = l["labels"]

        return loss, sample_size, logging_output

    def compute_loss(
            self,
            name,
            model,
            net_output,
            target,
            weight,
            ignore_index=None,
            reduce=True,
            factor=1.0):
        if name == "score":
            loss = F.mse_loss(net_output, target, reduction="mean")
            loss = loss * factor
            return {"name": name, "loss": loss, "factor": factor}

        else:
            lprobs = model.get_normalized_probs(net_output, log_probs=True)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs,
                target,
                self.eps,
                weight,
                ignore_index,
                reduce,
            )
            # reduce by batch size, same as score
            # loss = loss * factor / target.size(0)
            loss = loss * factor
            return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )

        for key in logging_outputs[0]:
            if key == "score_loss":
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key,
                    val / sample_size,
                    sample_size,
                    round=6,
                )
            elif key[-5:] == "_loss":
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key,
                    val / sample_size if sample_size > 0 else 0.0,
                    sample_size,
                    round=6,
                )
            elif key == "score_labels":
                all_y = []
                all_y_hat = []
                for log in logging_outputs:
                    labels = log.get(key, None)
                    if labels is not None:
                        y = labels.get("y", None)
                        if y is not None:
                            all_y.append(y)
                        y_hat = labels.get("y_hat", None)
                        if y_hat is not None:
                            all_y_hat.append(y_hat)
                y = torch.cat(all_y)
                y_hat = torch.cat(all_y_hat)

                # The first char of key must be "_".
                log_labels("_score_labels", [y, y_hat])
                metrics.log_derived("pearson", compute_pearsonr)

            elif key[-7:] == "_labels":
                all_y = []
                all_y_hat = []
                for log in logging_outputs:
                    labels = log.get(key, None)
                    if labels is not None:
                        y = labels.get("y", None)
                        if y is not None:
                            all_y.append(y)
                        y_hat = labels.get("y_hat", None)
                        if y_hat is not None:
                            all_y_hat.append(y_hat)
                y = torch.cat(all_y)
                y_hat = torch.cat(all_y_hat)
                log_labels("_"+key[:-7]+"_labels", [y, y_hat])
                metrics.log_derived(key[:-7] + "_f1_mult", get_f1_compute_fn(key[:-7], "mult"))
                metrics.log_derived(key[:-7] + "_f1_ok", get_f1_compute_fn(key[:-7], "ok"))
                metrics.log_derived(key[:-7] + "_f1_bad", get_f1_compute_fn(key[:-7], "bad"))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
