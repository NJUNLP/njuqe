# Copyright (c) 2020 the NJUQE authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from fairseq import utils
from fairseq.models.fairseq_model import BaseFairseqModel
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerModel
from njuqe.modules.qe_detector import QEDetector
logger = logging.getLogger(__name__)


class Generator(TransformerModel):
    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            return_all_hiddens: bool = True,
            features_only: bool = False,
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


@register_model("direct_qe")
class DirectQE(BaseFairseqModel):

    def __init__(self, args, task, generator):
        super().__init__()

        self.task = task
        self.tgt_dict = task.target_dictionary
        self.generator = generator
        self.train_generator = args.train_generator

        if not args.train_generator:
            self.tag_dict = task.tag_dictionary
            self.qe_detector = QEDetector(args, task)

    @staticmethod
    def add_args(parser):

        parser.add_argument('--train-generator', default=False, action='store_true',
                            help='train-generator')

        # generator的参数，就用原来的，看上去会整洁一点
        TransformerModel.add_args(parser)

        # detector-discriminator的参数，都增加了-extractor的后缀
        parser.add_argument('--activation-fn-extractor',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout-extractor', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout-extractor', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout-extractor', '--relu-dropout-extractor', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path-extractor', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim-extractor', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim-extractor', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers-extractor', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads-extractor', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before-extractor', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos-extractor', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path-extractor', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim-extractor', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim-extractor', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers-extractor', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads-extractor', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos-extractor', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before-extractor', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--decoder-output-dim-extractor', type=int, metavar='N',
                            help='decoder output dimension (extra linear layer '
                                 'if different from decoder embed dim')
        parser.add_argument('--share-decoder-input-output-embed-extractor', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings-extractor', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings-extractor', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff-extractor', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout-extractor', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--layernorm-embedding-extractor', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding-extractor', action='store_true',
                            help='if True, dont scale embeddings')
        parser.add_argument('--checkpoint-activations-extractor', action='store_true',
                            help='checkpoint activations at each layer, which saves GPU '
                                 'memory usage at the cost of some additional compute')
        parser.add_argument('--no-cross-attention-extractor', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention-extractor', default=False, action='store_true',
                            help='perform cross+self-attention')
        parser.add_argument('--encoder-layerdrop-extractor', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop-extractor', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep-extractor', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep-extractor', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--quant-noise-pq-extractor', type=float, metavar='D', default=0,
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size-extractor', type=int, metavar='D', default=8,
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar-extractor', type=float, metavar='D', default=0,
                            help='scalar quantization noise and scalar quantization at training time')

        # qe-fine-tune部分的参数
        parser.add_argument('--dropout-qe', type=float, metavar='D',
                            help='dropout probability')

        # qe-fine-tune部分的参数
        parser.add_argument('--load-embed', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)
        generator = Generator.build_model(args, task)
        return cls(args, task, generator)

    def get_masked_data(self, y):
        lm_ids = torch.full_like(y, self.tgt_dict.pad())
        y_mask = y.clone()
        for i, sentence in enumerate(y):
            for j, word in enumerate(sentence):
                prob = torch.rand(1)
                if prob < 0.15:
                    prob = prob / 0.15
                    if prob < 100:
                        y_mask[i][j] = self.tgt_dict.index("<mask>")
                    lm_ids[i][j] = y[i][j]
        # 返回：（1）含有mask标签的y （2）被mask部分是原id，其余用tgt_dict.pad()填充的lm_ids
        return y_mask, lm_ids

    def get_noise_data(self, gold, pre, lm_ids, k=10):
        # 此函数借鉴自cuiq的代码
        # 由于是自己生成伪数据，没有输入tag_dict的信息，就将pad为2，ok为0，bad为1
        pre = pre.topk(k=k, dim=2)
        batch_size, seq_len = pre[1][:, :, 0].size()

        pre = pre[1]

        lm_ids = lm_ids.tolist()
        gold = gold.tolist()
        noise_data = []
        noise_label = []
        hter_label = []

        ones_token = torch.ones(k)

        for each_sen_1st, each_sen_gold, each_sen_lm in zip(pre, gold, lm_ids):
            bad_count = 0.0 + 1e-12
            all_count = 0.0 + 1e-12

            pad_idx = self.tag_dict.index("PAD") - self.tag_dict.nspecial
            ok_idx = self.tag_dict.index("OK") - self.tag_dict.nspecial
            bad_idx = self.tag_dict.index("BAD") - self.tag_dict.nspecial
            for each_1st, each_gold, each_id in zip(each_sen_1st, each_sen_gold, each_sen_lm):
                if each_gold == self.tgt_dict.bos():
                    noise_data.append(self.tgt_dict.bos())
                    noise_label.append(pad_idx)
                    noise_data.append(self.tgt_dict.index("<score>"))
                    noise_label.append(pad_idx)
                elif each_gold == self.tgt_dict.pad() or each_gold == self.tgt_dict.eos():
                    noise_data.append(each_gold)
                    noise_label.append(pad_idx)
                else:
                    all_count += 1
                    choose_one = each_1st[ones_token.multinomial(num_samples=1, replacement=True)]
                    if each_id != self.tgt_dict.pad() and choose_one != each_id:
                        noise_data.append(choose_one)
                        bad_count += 1
                        noise_label.append(bad_idx)
                    else:
                        noise_data.append(each_gold)
                        noise_label.append(ok_idx)
            hter_label.append(bad_count / all_count)

        noise_data = torch.tensor(noise_data).view((batch_size, seq_len+1))
        noise_label = torch.tensor(noise_label).view((batch_size, seq_len+1))
        hter_label = torch.tensor(hter_label).view((batch_size, 1))

        return noise_data, noise_label, hter_label

    def forward(
        self,
        src_tokens=None,
        mt_tokens=None,
        fine_tune=False,
        generate=False,
    ):
        if self.train_generator:
            if self.training:
                mt_tokens_mask, mt_tokens_lmids = self.get_masked_data(mt_tokens)           # 随机mask数据
                mt_output = self.generator(src_tokens, None, mt_tokens_mask)[0]             # 将包括mask的数据传入transformer训练generator
                outputs = {
                    "mt_tag":{
                        "out": mt_output,
                        "tgt": mt_tokens_lmids,
                    }
                }
            else:
                mt_output = self.generator(src_tokens, None, mt_tokens)[0]
                outputs = {
                    "mt_tag":{
                        "out": mt_output,
                        "tgt": mt_tokens,
                    }
                }
            return outputs
        else:
            # TODO 是否可以换成utils.move_to_cuda
            device = next(self.parameters()).device
            mt_noise_label, mt_hter_label = None, None
            if not fine_tune:
                if not self.training:
                    torch.manual_seed(1234)
                with torch.no_grad():
                    mt_tokens_mask, mt_tokens_lmids = self.get_masked_data(mt_tokens)   # 首先获取masked-data
                    mt_output = self.generator(src_tokens, None, mt_tokens_mask)[0]     # 得到mask词的logits，下一步生成pseudo-data
                mt_noise_data, mt_noise_label, mt_hter_label = self.get_noise_data(mt_tokens, mt_output, mt_tokens_lmids)
                mt_tokens = mt_noise_data.to(device) if mt_noise_data is not None else None
            qe_output = {}
            if generate:
                qe_output["src_tokens"] = src_tokens
                qe_output["mt_tokens"] = mt_tokens
                qe_output["mt_tag"] = mt_noise_label.to(device) if mt_noise_label is not None else None
                qe_output["score"] = mt_hter_label.to(device) if mt_hter_label is not None else None
            else:
                qe_output = self.qe_detector(src_tokens, mt_tokens)
                if "mt_tag" in qe_output:
                    qe_output["mt_tag"]["tgt"] = mt_noise_label.to(device) if mt_noise_label is not None else None
                if "score" in qe_output:
                    qe_output["score"]["tgt"] = mt_hter_label.to(device) if mt_hter_label is not None else None
            return qe_output

    def get_normalized_probs_scriptable(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
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
        raise NotImplementedError

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        super().upgrade_state_dict_named(state_dict, name)

        # 把所有qe部分写在qe_detector中
        if hasattr(self, "qe_detector"):
            cur_state = self.qe_detector.state_dict()
            for k, v in cur_state.items():
                if prefix + "qe_detector." + k not in state_dict:
                    logger.info("Overwriting " + prefix + "qe_detector." + k)
                    state_dict[prefix + "qe_detector." + k] = v

        if hasattr(self, "qe_detector") and (not self.task.args.fine_tune or self.task.args.load_embed): # 也就是train-discriminator的时候要加载train-mlm的词嵌入初始化
            cur_encoder_embed_state = self.generator.encoder.embed_tokens.state_dict()
            cur_decoder_embed_state = self.generator.decoder.embed_tokens.state_dict()
            times_bigger = int(self.qe_detector.extractor.args.decoder_output_dim / self.generator.args.decoder_output_dim)
            for k, v in cur_encoder_embed_state.items():
                self.qe_detector.extractor.encoder.embed_tokens.state_dict()[k].copy_(v.repeat(1, times_bigger))
            for k, v in cur_decoder_embed_state.items():
                self.qe_detector.extractor.decoder.embed_tokens.state_dict()[k].copy_(v.repeat(1, times_bigger))

    def get_targets(self, sample, net_output):
        return net_output["mt_tag_tgt"]     # 这边是为了generator时获取的仅仅是被mask的，没被mask的都是padding，从而算mask-loss


@register_model_architecture("direct_qe", "direct_qe")
def base_architecture(args):
    args.dropout_qe = getattr(args, "dropout_qe", 0.1)

    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_embed_dim_extractor = getattr(args, "encoder_embed_dim_extractor", 512)