# Copyright (c) 2020 the NJUQE authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models.lstm import LSTM


class QEHead(nn.Module):
    """Head for QE tasks."""

    def __init__(
            self,
            predict_target,
            predict_source,
            predict_gaps,
            predict_score,
            hidden_size,
            sent_pooling,
            num_classes=2,
            pooler_dropout=0.0,
            activation_fn=nn.Sigmoid(),
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.predict_target = predict_target
        self.predict_source = predict_source
        self.predict_gaps = predict_gaps
        self.predict_score = predict_score
        self.activation_fn = activation_fn

        self.mt_dense = nn.Linear(hidden_size, num_classes) if self.predict_target else None
        self.src_dense = nn.Linear(hidden_size, num_classes) if self.predict_source else None
        self.gap_dense = nn.Linear(hidden_size, num_classes) if self.predict_gaps else None
        if sent_pooling == "mixed":
            self.score_dense = nn.Linear(hidden_size*2, 1) if self.predict_score else None
        else:
            self.score_dense = nn.Linear(hidden_size, 1) if self.predict_score else None

    def forward(self, features):
        outputs = {}
        if self.predict_target:
            feature_target = features.get("mt_tag", None)
            if feature_target is None:
                raise Exception(
                    "Model didn't output features for mt_tag!"
                )
            x = self.dropout(feature_target)
            x = self.mt_dense(x)
            outputs["mt_tag"] = {"out": x}

        if self.predict_source:
            feature_source = features.get("src_tag", None)
            if feature_source is None:
                raise Exception(
                    "Model didn't output features for src_tag!"
                )
            x = self.dropout(feature_source)
            x = self.src_dense(x)
            outputs["src_tag"] = {"out": x}

        if self.predict_gaps:
            feature_gaps = features.get("mt_gap_tag", None)
            if feature_gaps is None:
                raise Exception(
                    "Model didn't output features for mt_gap_tag!"
                )
            x = self.dropout(feature_gaps)
            x = self.gap_dense(x)
            outputs["mt_gap_tag"] = {"out": x}

        if self.predict_score:
            feature_score = features.get("score", None)
            if feature_score is None:
                raise Exception(
                    "Model didn't output features for score!"
                )
            x = self.dropout(feature_score)
            x = self.score_dense(x)
            x = self.activation_fn(x)
            outputs["score"] = {"out": x}

        return outputs

# class QEHead(nn.Module):
#     """Head for QE tasks."""
#
#     def __init__(
#             self,
#             predict_target,
#             predict_source,
#             predict_gaps,
#             predict_score,
#             embed_dim,
#             hidden_size,
#             num_classes=2,
#             pooler_dropout=0.0,
#             bidirectional=True,
#     ):
#         super().__init__()
#         self.dropout = nn.Dropout(p=pooler_dropout)
#         if predict_target:
#             self.lstm = LSTM(
#                 input_size=embed_dim,
#                 hidden_size=hidden_size,
#                 bidirectional=bidirectional,
#             )
#             if bidirectional:
#                 hidden_size *= 2
#             self.dense = nn.Linear(hidden_size, num_classes)
#
#     def forward(self, features):
#         outputs = {
#             "mt_tag": {
#                 "out": None,
#             },
#             "mt_gap_tag": {
#                 "out": None,
#             },
#             "src_tag": {
#                 "out": None,
#             },
#             "score": {
#                 "out": None,
#             },
#         }
#         return outputs
