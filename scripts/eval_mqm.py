import logging
import os
import sys
from argparse import Namespace
from itertools import chain

import torch
from fairseq import checkpoint_utils, distributed_utils, options, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import metrics, progress_bar
from omegaconf import DictConfig


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.validate")


def main(cfg: DictConfig, override_args=None):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

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

    # Build criterion
    criterion = task.build_criterion(saved_cfg.criterion)
    criterion.eval()

    try:
        test_subset = cfg.task.eval_split
        task.load_dataset(test_subset, combine=False, epoch=1, task_cfg=saved_cfg.task)
        dataset = task.dataset(test_subset)
    except KeyError:
        raise Exception("Cannot find dataset: " + test_subset)

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
        prefix=f"test on '{test_subset}' subset",
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    log_outputs = []
    score_file = cfg.task.qe_destdir + "/{}.mqm_score".format(test_subset)   # 生成的文件存放位置，可以写的更好看一点，比如从args获取
    tags_file = cfg.task.qe_destdir + "/{}.tags".format(test_subset)
    score_file = open(score_file, 'w')
    tags_file = open(tags_file, 'w')

    for i, sample in enumerate(progress):
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        with torch.no_grad():
            _loss, _sample_size, log_output = task.valid_step(sample, model, criterion)
        progress.log(log_output, step=i)
        # 暂时是mt_tag和score，其他的类似加就可以了
        if "mt_tag_labels" in log_output:
            tags_gold = sample["mt_tag"]
            tags_hat = log_output["mt_tag_labels"]["y_hat"]
            index = 0
            for gold in tags_gold:
                gold_len = gold.ne(task.tag_dict.index("PAD") - task.tag_dict.nspecial).sum()
                pred_str = ""
                for i in range(gold_len):
                    print(task.tag_dict[4], task.tag_dict[5], task.tag_dict[6])
                    pred_str += task.tag_dict[tags_hat[index].item() + task.tag_dict.nspecial] + " "
                    index += 1
                tags_file.write(pred_str + '\n')
        if "score_labels" in log_output:
            score_hat = log_output["score_labels"]["y_hat"]
            for item in score_hat:
                score_file.write(str(item.item()) + '\n')
        log_outputs.append(log_output)

    if cfg.distributed_training.distributed_world_size > 1:
        log_outputs = distributed_utils.all_gather_list(
            log_outputs,
            max_size=cfg.common.all_gather_list_size,
            group=distributed_utils.get_data_parallel_group(),
        )
        log_outputs = list(chain.from_iterable(log_outputs))

    with metrics.aggregate() as agg:
        task.reduce_metrics(log_outputs, criterion)
        log_output = agg.get_smoothed_values()

    progress.print(log_output, tag=test_subset, step=i)


def cli_main():
    parser = options.get_validation_parser()
    args = options.parse_args_and_arch(parser)

    # only override args that are explicitly given on the command line
    override_parser = options.get_validation_parser()
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)

    distributed_utils.call_main(convert_namespace_to_omegaconf(args), main, override_args=override_args)


if __name__ == "__main__":
    cli_main()
