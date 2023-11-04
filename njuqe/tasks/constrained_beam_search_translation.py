import logging
import math
import torch
from typing import Dict, List, Optional
from torch import Tensor
from fairseq.tasks.translation import TranslationTask
from fairseq.tasks import register_task
from fairseq.search import BeamSearch
from fairseq.sequence_generator import SequenceGenerator

logger = logging.getLogger(__name__)


class ConstrainedSequenceGenerator(SequenceGenerator):

    def __init__(
            self,
            models,
            tgt_dict,
            beam_size=1,
            max_len_a=0,
            max_len_b=200,
            min_len=1,
            normalize_scores=True,
            len_penalty=1.0,
            unk_penalty=0.0,
            temperature=1.0,
            match_source_len=False,
            no_repeat_ngram_size=0,
            search_strategy=None,
            eos=None,
            symbols_to_strip_from_output=None,
            lm_model=None,
            lm_weight=1.0,
            window_max_delete=3,
            window_max_insert=3,
            threshold_prob=0.1,
            lamda_ratio=0.55,
            softmax_temperature=0.20,
    ):
        super().__init__(models, tgt_dict, beam_size, max_len_a, max_len_b, min_len, normalize_scores,
                         len_penalty, unk_penalty, temperature, match_source_len, no_repeat_ngram_size,
                         search_strategy, eos, symbols_to_strip_from_output, lm_model, lm_weight)

        self.window_max_delete = window_max_delete
        self.window_max_insert = window_max_insert
        self.threshold_prob = threshold_prob
        self.lamda_ratio = lamda_ratio
        self.softmax_temperature = softmax_temperature

    def _generate(
            self,
            sample: Dict[str, Dict[str, Tensor]],
            prefix_tokens: Optional[Tensor] = None,
            constraints: Optional[Tensor] = None,
            bos_token: Optional[int] = None,
    ):
        incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(self.model.models_size)
            ],
        )
        net_input = sample["net_input"]

        if "src_tokens" in net_input:
            src_tokens = net_input["src_tokens"]
            # length of the source text being the character length except EndOfSentence and pad
            src_lengths = (
                (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
            )
        elif "source" in net_input:
            src_tokens = net_input["source"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        else:
            raise Exception("expected src_tokens or source in net input")

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimenions (i.e. audio features)
        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size

        if constraints is not None and not self.search.supports_constraints:
            raise NotImplementedError(
                "Target-side constraints were provided, but search method doesn't support them"
            )

        # Initialize constraints, when active
        self.search.init_constraints(constraints, beam_size)

        max_len: int = -1
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                self.model.max_decoder_positions() - 1,
            )
        assert (
                self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"
        # compute the encoder output for each beam
        encoder_outs = self.model.forward_encoder(net_input)

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)
        # ensure encoder_outs is a List.
        assert encoder_outs is not None

        # initialize buffers
        scores = (
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        )  # +1 for eos; pad is never chosen for scoring
        tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
                .to(src_tokens)
                .long()
                .fill_(self.pad)
        )  # +2 for eos and pad
        tokens[:, 0] = self.eos if bos_token is None else bos_token
        attn: Optional[Tensor] = None

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = (
            torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        finished = [
            False for i in range(bsz)
        ]  # a boolean array indicating if the sentence at the index is finished or not
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None

        original_batch_idxs: Optional[Tensor] = None
        if "id" in sample and isinstance(sample["id"], Tensor):
            original_batch_idxs = sample["id"]
        else:
            original_batch_idxs = torch.arange(0, bsz).type_as(tokens)

        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                        batch_idxs
                    )
                    reorder_state.view(-1, beam_size).add_(
                        corr.unsqueeze(-1) * beam_size
                    )
                    original_batch_idxs = original_batch_idxs[batch_idxs]
                self.model.reorder_incremental_state(incremental_states, reorder_state)
                encoder_outs = self.model.reorder_encoder_out(
                    encoder_outs, reorder_state
                )

            lprobs, avg_attn_scores = self.model.forward_decoder(
                tokens[:, : step + 1],
                encoder_outs,
                incremental_states,
                self.temperature,
            )

            if self.lm_model is not None:
                lm_out = self.lm_model(tokens[:, : step + 1])
                probs = self.lm_model.get_normalized_probs(
                    lm_out, log_probs=True, sample=None
                )
                probs = probs[:, -1, :] * self.lm_weight
                lprobs += probs

            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # handle max length constraint
            if step >= max_len:
                lprobs[:, : self.eos] = -math.inf
                lprobs[:, self.eos + 1:] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            if (
                    prefix_tokens is not None
                    and step < prefix_tokens.size(1)
                    and step < max_len
            ):
                lprobs, tokens, scores = self._prefix_tokens(
                    step, lprobs, scores, tokens, prefix_tokens, beam_size
                )
            elif step < self.min_len:
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, self.eos] = -math.inf

            # Record attention scores, only support avg_attn_scores is a Tensor
            if avg_attn_scores is not None:
                if attn is None:
                    attn = torch.empty(
                        bsz * beam_size, avg_attn_scores.size(1), max_len + 2
                    ).to(scores)
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)
            eos_bbsz_idx = torch.empty(0).to(
                tokens
            )  # indices of hypothesis ending with eos (finished sentences)
            eos_scores = torch.empty(0).to(
                scores
            )  # scores of hypothesis ending with eos (finished sentences)

            if self.should_set_src_lengths:
                self.search.set_src_lengths(src_lengths)

            if self.no_repeat_ngram_size > 0:
                lprobs = self._no_repeat_ngram(tokens, lprobs, bsz, beam_size, step)

            # Shape: (batch, cand_size)
            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
                sample,
                self.window_max_delete,
                self.window_max_insert,
                self.threshold_prob,
                self.lamda_ratio,
                self.softmax_temperature,
                tokens[:, : step + 1],
                original_batch_idxs,
            )

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            # Shape of eos_mask: (batch size, beam size)
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)

            # only consider eos when it's among the top beam_size indices
            # Now we know what beam item(s) to finish
            # Shape: 1d list of absolute-numbered
            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
            )

            finalized_sents: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
                )

                finalized_sents = self.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    scores,
                    finalized,
                    finished,
                    beam_size,
                    attn,
                    src_lengths,
                    max_len,
                )
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            if self.search.stop_on_max_len and step >= max_len:
                break

            # if step >= max_len:
            #     print(sample)
            #     print(original_batch_idxs)
            #     print(tokens)

            assert step < max_len, f"{step} < {max_len}"

            # Remove finalized sentences (ones for which {beam_size}
            # finished hypotheses have been generated) from the batch.
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(
                    bsz, dtype=torch.bool, device=cand_indices.device
                )
                batch_mask[finalized_sents] = False
                # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
                batch_idxs = torch.arange(
                    bsz, device=cand_indices.device
                ).masked_select(batch_mask)

                # Choose the subset of the hypothesized constraints that will continue
                self.search.prune_sentences(batch_idxs)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]

                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, attn.size(1), -1
                    )
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in torchscript.

            eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just
            # the hypos with the smallest values in active_mask.
            # {active_hypos} indicates which {beam_size} hypotheses
            # from the list of {2 * beam_size} candidates were
            # selected. Shapes: (batch size, beam size)
            new_cands_to_ignore, active_hypos = torch.topk(
                active_mask, k=beam_size, dim=1, largest=False
            )

            # update cands_to_ignore to ignore any finalized hypos.
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            # Make sure there is at least one active item for each sentence in the batch.
            assert (~cands_to_ignore).any(dim=1).all()

            # update cands_to_ignore to ignore any finalized hypos

            # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
            # can be selected more than once).
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses

            # Set the tokens for each beam (can select the same row more than once)
            tokens[:, : step + 1] = torch.index_select(
                tokens[:, : step + 1], dim=0, index=active_bbsz_idx
            )
            # Select the next token for each of them
            tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(
                cand_indices, dim=1, index=active_hypos
            )
            if step > 0:
                scores[:, :step] = torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx
                )
            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )

            # Update constraints based on which candidates were selected for the next beam
            self.search.update_constraints(active_hypos)

            # copy attention for active hypotheses
            if attn is not None:
                attn[:, :, : step + 2] = torch.index_select(
                    attn[:, :, : step + 2], dim=0, index=active_bbsz_idx
                )

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            scores = torch.tensor(
                [float(elem["score"].item()) for elem in finalized[sent]]
            )
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = torch.jit.annotate(
                List[Dict[str, Tensor]], finalized[sent]
            )
        return finalized


class ConstrainedBeamSearch(BeamSearch):

    def __init__(self, tgt_dict):
        super().__init__(tgt_dict)
        self.stop_on_max_len = True
        self.constraint_states = None
        # 表示当前到对应的ref语句的step
        self.step_ref = []
        # 表示当前连续发生了多少次的修改
        self.continuous_changes = []
        # 表示当前一共发生了多少次的修改
        self.all_changes = []

    @torch.jit.export
    def step(
            self,
            step: int,
            lprobs,
            scores: Optional[Tensor],
            sample,
            window_max_delete,
            window_max_insert,
            threshold_prob,
            lamda_ratio,
            softmax_temperature,
            prev_output_tokens: Optional[Tensor] = None,
            original_batch_idxs: Optional[Tensor] = None,
    ):

        # 记录基础的信息，是原代码中的内容
        bsz, beam_size, vocab_size = lprobs.size()

        # 初始化记录的变量信息，必须放在前面
        if step == 0:
            self.step_ref = torch.full([bsz, beam_size * 2], 0, dtype=torch.int64).cuda()
            self.continuous_changes = torch.full([bsz, beam_size * 2], 0, dtype=torch.int64).cuda()
            self.all_changes = torch.full([bsz, beam_size * 2], 0, dtype=torch.int64).cuda()

        # 因为bsz中部分语句可能结束了，所以要选择，语句的目的是找到ref语句
        bsz_select_idx = torch.zeros_like(sample['id'])
        for idx in range(len(sample['id'])):
            if sample['id'][idx] in original_batch_idxs:
                bsz_select_idx[idx] = 1
        bsz_select = (bsz_select_idx > 0).cuda()
        ref_token = sample['target'][bsz_select].cuda()

        # 这边要进行概率的调整，根据距离调整，调整策略有多种，这是其中一种，可以按需修改
        distance_strategy = 1

        if distance_strategy:

            softmax_temperature = torch.tensor(softmax_temperature, dtype=torch.float64).cuda()
            lamda_ratio = torch.tensor(lamda_ratio, dtype=torch.float64).cuda()
            threshold_prob = torch.tensor(threshold_prob, dtype=torch.float64).cuda()
            max_value = torch.tensor(5e3, dtype=torch.float64).cuda()

            # 距离函数可以有多种，这是其中一种，可以按需调整
            def distance_cal(dis):
                return (1.0 / dis / dis) * 1.0

            # 如果step-ref超过范围，让它尽量对准bos
            bos_position = torch.where(
                ref_token == 2, torch.arange(0, ref_token.size(1), 1).cuda(), torch.tensor(0).cuda()
            ).sum(dim=-1).unsqueeze(-1).repeat(1, beam_size * 2).cuda()

            # 这是因为后面会超出eos到pad，甚至超过pad，但是我们不能让他指向eos，那样-continuous就会出问题，所以有一个copy的副本
            step_ref_exceed_copy = torch.where(
                self.step_ref[bsz_select] > bos_position, bos_position, self.step_ref[bsz_select]
            )
            # 因为ref是有指向边界的最大值的，但此时continuous—change不能无止尽的增大，所以+1的地方要-1
            self.continuous_changes[bsz_select] = torch.where(
                self.step_ref[bsz_select] >= ref_token.size(1), self.continuous_changes[bsz_select] - 1,
                self.continuous_changes[bsz_select]
            )
            # 让超出范围的对准最后一个，不是eos就是pad
            self.step_ref[bsz_select] = torch.where(
                self.step_ref[bsz_select] >= ref_token.size(1),
                torch.tensor(ref_token.size(1) - 1, dtype=torch.int64).cuda(), self.step_ref[bsz_select]
            )

            weight_add_cur = torch.arange(1, ref_token.size(1) + 1, 1).repeat(bsz, beam_size,
                                                                              1).double().cuda()  # 用来表示增加的权重，初始化是 1 2 ... vocab，形状是[bsz,beamsize,vocabsize]
            ref_step_expend = step_ref_exceed_copy[:, :beam_size].unsqueeze(-1).repeat(1, 1, ref_token.size(
                1)).cuda()  # 将step_ref扩展，为了便于下一步的计算
            weight_add_cur = torch.where((weight_add_cur <= ref_step_expend), max_value,
                                         weight_add_cur)  # 当前refstep前的词，也就是已经翻译过的词权重add都设置为max_value
            weight_add_cur = weight_add_cur - (weight_add_cur.min(dim=-1)[0] - 1).unsqueeze(-1).repeat(1, 1,
                                                                                                       ref_token.size(
                                                                                                           1))  # 待翻译的词每个都从1开始
            weight_add_cur = distance_cal(weight_add_cur)
            weight_add_cur = torch.where(weight_add_cur < distance_cal(max_value - 1000),
                                         torch.tensor(0.0, dtype=torch.float64).cuda(),
                                         weight_add_cur)  # 把前面已经翻译的部分都转化为0，得到最后要加的weight-add

            weight_add_pre = torch.arange(1, ref_token.size(1) + 1, 1).repeat(bsz, beam_size, 1).double().cuda()
            pre_ref = self.step_ref[bsz_select] - self.continuous_changes[bsz_select]
            ref_step_expend = pre_ref[:, :beam_size].unsqueeze(-1).repeat(1, 1, ref_token.size(1)).cuda()
            weight_add_pre = torch.where((weight_add_pre <= ref_step_expend), max_value, weight_add_pre)
            weight_add_pre = weight_add_pre - (weight_add_pre.min(dim=-1)[0] - 1).unsqueeze(-1).repeat(1, 1,
                                                                                                       ref_token.size(
                                                                                                           1))
            weight_add_pre = distance_cal(weight_add_pre)
            weight_add_pre = torch.where(weight_add_pre < distance_cal(max_value - 1000),
                                         torch.tensor(0.0, dtype=torch.float64).cuda(), weight_add_pre)

            weight_add_comprehensive = (weight_add_cur + weight_add_pre) / 2.0  # 综合的权重，两者相加

            weight = torch.zeros_like(lprobs)  # 为bsz，beamsize，vocab的大小的内容加上一个权限，这是初始化
            ref_token_expand = ref_token.unsqueeze(1).repeat(1, beam_size, 1)
            weight.scatter_add_(-1, ref_token_expand, weight_add_comprehensive.float())  # 直接把权重加上去
            weight = torch.where(weight == 0.0, torch.tensor(-math.inf, dtype=torch.float64).cuda(), weight.double())
            weight[:, :, self.pad] = -math.inf
            weight = torch.nn.functional.softmax(weight / softmax_temperature, dim=2)

            probs_nmt = torch.exp(lprobs)
            ref_token_idx = torch.gather(ref_token, dim=1, index=self.step_ref[bsz_select][:, :beam_size])
            ref_probs = torch.gather(probs_nmt, dim=-1, index=ref_token_idx.unsqueeze(-1))
            ref_probs_over_threshold = (ref_probs > threshold_prob)

            probs = lamda_ratio * probs_nmt + (1 - lamda_ratio) * weight
            probs_res = torch.where(ref_probs_over_threshold, probs.float(), probs_nmt)

            lprobs = torch.log(probs_res)

            lprobs[:, :, self.pad] = -math.inf  # 永不生成pad
            if step == 0:
                lprobs[:, :, self.eos] = -math.inf  # 一开始没生成词的时候eos不能生成

        # 是beam search原本的代码，step=0是因为一开始所有beam一视同仁，后来只要相加即可
        if step == 0:
            lprobs = lprobs[:, ::beam_size, :].contiguous()
        else:
            assert scores is not None
            lprobs = lprobs + scores[:, :, step - 1].unsqueeze(-1)

        # 是beam search原本的代码，选取topk的候选
        top_prediction = torch.topk(
            lprobs.view(bsz, -1),
            k=min(
                beam_size * 2,
                lprobs.view(bsz, -1).size(1) - 1,
            ),
        )
        scores_buf = top_prediction[0]
        indices_buf = top_prediction[1]
        beams_buf = indices_buf // vocab_size
        indices_buf = indices_buf.fmod(vocab_size)

        # 这一轮是继承的哪个beam的变量，step=0的时候无所谓因为大家都相同
        self.step_ref[bsz_select] = torch.gather(self.step_ref[bsz_select], dim=1, index=beams_buf)
        self.continuous_changes[bsz_select] = torch.gather(self.continuous_changes[bsz_select], dim=1, index=beams_buf)
        self.all_changes[bsz_select] = torch.gather(self.all_changes[bsz_select], dim=1, index=beams_buf)

        # 当前step步ref应当的词
        ref_step = torch.gather(ref_token, dim=1, index=self.step_ref[bsz_select])

        # 看当前的选择indices和ref是否相同
        ref_match = (indices_buf == ref_step)

        # 所有的step都要加一
        self.step_ref[bsz_select] += 1

        # 如果和ref一致，那么continuous_change清零
        self.continuous_changes[bsz_select] = torch.where(ref_match, torch.tensor(0, dtype=torch.int64).cuda(),
                                                          self.continuous_changes[bsz_select])

        # 如果不和ref一致，那么（1）continuous_change加一，（2）all_change加一
        self.continuous_changes[bsz_select] += (~ref_match)
        self.all_changes[bsz_select] += (~ref_match)

        # 不一致，替换，没有任何操作
        pass

        # 不一致，插入了非ref中的词，和前面的词匹配
        for window_size in range(window_max_insert):
            cur_word = torch.where(self.continuous_changes[bsz_select] > (1 + window_size), indices_buf,
                                   torch.tensor(-1, dtype=torch.int64).cuda())  # 找到连续修改大于一定范围的词
            if cur_word.sum() != ((-1) * cur_word.size(0) * cur_word.size(1)):  # 存在连续修改的词
                prev_ref = self.step_ref[bsz_select] - window_size - 2  # window=0，就是考虑前一个词
                prev_ref = torch.where(prev_ref < 0, torch.tensor(0, dtype=torch.int64).cuda(),
                                       prev_ref)  # 可能超出范围，那样用第一个词，也在比较范围内，其实用什么都一样，因为如果超出范围，那么该词必定为-1，不会匹配上词的
                prev_word = torch.gather(ref_token, dim=1, index=prev_ref)  # 找到前面对应的词
                prev_word_match = (cur_word == prev_word)
                # 找到匹配的词，那么（1）修改对应的step，（2）continuous_change为0，（3）all_change减一
                self.step_ref[bsz_select] = torch.where(prev_word_match, self.step_ref[bsz_select] - window_size - 1,
                                                        self.step_ref[bsz_select])
                self.continuous_changes[bsz_select] = torch.where(prev_word_match,
                                                                  torch.tensor(0, dtype=torch.int64).cuda(),
                                                                  self.continuous_changes[bsz_select])
                self.all_changes[bsz_select] -= prev_word_match.int()

        # 不一致，删除了ref中的词，所以和后面的词匹配
        for window_size in range(window_max_delete):
            cur_word = torch.where(self.continuous_changes[bsz_select].bool(), indices_buf,
                                   torch.tensor(-1, dtype=torch.int64).cuda())  # self.continuous_changes不为0的词
            if cur_word.sum() != ((-1) * cur_word.size(0) * cur_word.size(1)):  # 存在连续修改的词
                next_ref = self.step_ref[bsz_select] + window_size  # window=0，就是考虑删除一个词
                next_ref = torch.where(next_ref >= ref_token.size(1),
                                       torch.tensor(ref_token.size(1) - 1, dtype=torch.int64).cuda(),
                                       next_ref)  # 可能会超出范围
                next_word = torch.gather(ref_token, dim=1, index=next_ref)  # 找到下一个词对应的词
                next_word_match = (cur_word == next_word)  # 只有self.continuous_changes不为0的词去匹配
                # 找到匹配的词，那么（1）修改对应的step，（2）continuous_change为0，（3）all_change减一
                self.step_ref[bsz_select] = torch.where(next_word_match, self.step_ref[bsz_select] + window_size + 1,
                                                        self.step_ref[bsz_select])
                self.continuous_changes[bsz_select] = torch.where(next_word_match,
                                                                  torch.tensor(0, dtype=torch.int64).cuda(),
                                                                  self.continuous_changes[bsz_select])

        initial_order = torch.arange(0, beam_size * 2, 1).repeat(bsz, 1).cuda()
        change_order = torch.where(indices_buf == self.eos, initial_order + beam_size * 2, initial_order)
        new_order = torch.sort(change_order, descending=False)[1]

        self.continuous_changes[bsz_select] = torch.gather(self.continuous_changes[bsz_select], 1, new_order)
        self.step_ref[bsz_select] = torch.gather(self.step_ref[bsz_select], 1, new_order)
        self.all_changes[bsz_select] = torch.gather(self.all_changes[bsz_select], 1, new_order)

        return scores_buf, indices_buf, beams_buf


@register_task("cbs_translation")
class ConstrainedBeamSearchTranslationTask(TranslationTask):

    @staticmethod
    def add_args(parser):
        TranslationTask.add_args(parser)
        parser.add_argument('--window-max-delete', default=3, type=int,
                            help='max window size to delete')
        parser.add_argument('--window-max-insert', default=3, type=int,
                            help='max window size to insert')
        parser.add_argument('--threshold-prob', default=0.10, type=float,
                            help='threshold prob')
        parser.add_argument('--lamda-ratio', default=0.55, type=float,
                            help='lamda ratio')
        parser.add_argument('--softmax-temperature', default=0.20, type=float,
                            help='softmax temperature')

    def build_generator(
            self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None
    ):
        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                compute_alignment=getattr(args, "print_alignment", False),
            )

        from fairseq.sequence_generator import (
            SequenceGenerator,
            SequenceGeneratorWithAlignment,
        )

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        constrained = getattr(args, "constraints", False)
        prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
        if (
                sum(
                    int(cond)
                    for cond in [
                        sampling,
                        diverse_beam_groups > 0,
                        match_source_len,
                        diversity_rate > 0,
                    ]
                )
                > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        search_strategy = ConstrainedBeamSearch(self.target_dictionary)

        if seq_gen_cls is None:
            if getattr(args, "print_alignment", False):
                seq_gen_cls = SequenceGeneratorWithAlignment
            else:
                seq_gen_cls = ConstrainedSequenceGenerator
        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
        return seq_gen_cls(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            window_max_delete=getattr(args, "window_max_delete", 3),
            window_max_insert=getattr(args, "window_max_insert", 3),
            threshold_prob=getattr(args, "threshold_prob", 0.10),
            lamda_ratio=getattr(args, "lamda_ratio", 0.55),
            softmax_temperature=getattr(args, "softmax_temperature", 0.20),
            **extra_gen_cls_kwargs,
        )
