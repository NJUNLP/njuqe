# Copyright (c) 2020 the NJUQE authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Dict, List, Optional, Tuple
import torch.nn as nn
from fairseq.models.transformer import TransformerModel
from njuqe.modules.qe_head import QEHead


class Extractor(TransformerModel):
    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            return_all_hiddens: bool = True,
            features_only: bool = True,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        # 变动：features_only: bool = True,  full_context_alignment=True，
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            full_context_alignment=True,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

class QEDetector(nn.Module):

    def __init__(
            self,
            args,
            task
    ):
        super().__init__()

        self.predict_target = args.predict_target
        self.predict_source = args.predict_source
        self.predict_gaps = args.predict_gaps
        self.predict_score = args.predict_score

        args_extractor = args_update(args)
        self.extractor = Extractor.build_model(args_extractor, task)

        self.qe_head = QEHead(
            self.predict_target,
            self.predict_source,
            self.predict_gaps,
            self.predict_score,
            args_extractor.decoder_embed_dim,
            pooler_dropout=args.dropout_qe,
        )

    def forward(self, src_tokens, mt_tokens):
        extractor_out = self.extractor(src_tokens, None, mt_tokens)[0]
        features = {}
        if self.predict_target:
            features["mt_tag"] = extractor_out
        if self.predict_score:
            features["score"] = extractor_out[:, 1]
        return self.qe_head(features)

def args_update(args):
    args_extractor = {}
    for k, v in vars(args).items():
        if k[-10:] == "_extractor":
            args_extractor[k[:-10]] = v
    args_extractor = argparse.Namespace(**args_extractor)
    return args_extractor