# Copyright (c) 2020 the NJUQE authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor
from fairseq import utils
from fairseq.models.fairseq_model import BaseFairseqModel
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerModel
from fairseq.models.roberta import XLMRModel, RobertaEncoder, RobertaClassificationHead
from fairseq.models.roberta import roberta_base_architecture
from fairseq.models.roberta import xlm_architecture
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import LayerNorm, TransformerSentenceEncoder
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from njuqe.modules.qe_head import QEHead
logger = logging.getLogger(__name__)


def select_positions(tensor, indices):
    range_vector = torch.arange(tensor.size(0), device=tensor.device).unsqueeze(1)
    return tensor[range_vector, indices]


@register_model("xlmr_qe")
class XLMRQE(XLMRModel):

    def __init__(self, task, args, encoder):
        XLMRModel.__init__(self, args, encoder)
        self.sent_pooling = args.sent_pooling
        self.tgt_dict = task.target_dictionary
        self.tag_dict = task.tag_dictionary
        if args.fine_tune:
            self.qe_head = QEHead(
                args.predict_target,
                args.predict_source,
                args.predict_gaps,
                args.predict_score,
                args.encoder_embed_dim,
                args.sent_pooling
            )

    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)
        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample
        encoder = RobertaEncoder(args, task.source_dictionary)
        return cls(task, args, encoder)

    def forward(
            self,
            src_tokens,
            features_only=False,
            return_all_hiddens=True,
            fine_tune=False,
            **kwargs
    ):
        features_only = fine_tune
        check_parallel = True if "joint_tokens" in kwargs else False

        input_tokens = kwargs["joint_tokens"] if check_parallel else src_tokens
        x, extra = self.encoder(input_tokens, features_only, return_all_hiddens, **kwargs)

        if not fine_tune:  # 使用普通数据调整模型
            return x, extra
        elif fine_tune:  # 使用真实qe数据微调
            features = {}
            if check_parallel:  # 取mt对应的部分出来
                mt = x[:, :kwargs["mt_tokens"].size(1)]
                masked = kwargs["masked_tokens_sing"][:, :kwargs["mt_tokens"].size(1)]
                if kwargs["bounds"] is not None:
                    mt = select_positions(mt, kwargs["bounds"])
                    masked = select_positions(masked, kwargs["bounds"])
            if self.args.predict_target:
                features["mt_tag"] = mt
            if self.args.predict_score:
                if self.sent_pooling == "mixed":
                    average_pooling = (mt * masked[:, :, None]).sum(1) / masked.sum(1)[:, None]
                    mixed_output = torch.cat([average_pooling, mt[:, 0]], 1)
                    features["score"] = mixed_output
                else:
                    features["score"] = mt[:, 0]  # TODO here follow RobertaClassificationHead score symbol ？
            outputs = self.qe_head(features)
            return outputs

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        super().upgrade_state_dict_named(state_dict, name)
        if hasattr(self, "qe_head"):
            cur_state = self.qe_head.state_dict()
            for k, v in cur_state.items():
                if prefix + "qe_head." + k not in state_dict:
                    logger.info("Overwriting " + prefix + "qe_head." + k)
                    state_dict[prefix + "qe_head." + k] = v

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        if torch.is_tensor(net_output):
            logits = net_output.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        else:
            logits = net_output[0].float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)


@register_model_architecture("xlmr_qe", "xlmr_qe")
def base_architecture(args):
    roberta_base_architecture(args)
    args.max_positions = getattr(args, "max_positions", 512)


@register_model_architecture("xlmr_qe", "xlmr_qe_large")
def base_architecture(args):
    args.max_positions = getattr(args, "max_positions", 512)
    args.encoder_layers = getattr(args, "encoder_layers", 24)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    roberta_base_architecture(args)