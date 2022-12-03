# NJUQE
[![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE) [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![Active](http://img.shields.io/badge/Status-Active-green.svg)](https://tterb.github.io) [![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)

NJUQE is an open-source toolkit to build machine translation quality estimation (QE) models.

## Requirements and Installation
* PyTorch version >= 1.5.0
* Python version >= 3.7
* cuda >= 10.1.243
* fairseq >=0.10.0

Quick Start:
```
cd $FAIRSEQ_PATH
wget https://github.com/facebookresearch/fairseq/archive/refs/tags/v0.10.0.tar.gz
tar -zxvf v0.10.0.tar.gz
cd fairseq-0.10.0
pip install --editable ./ 
python setup.py build_ext --inplace
cd $NJUQE_PATH
git clone https://github.com/NJUNLP/njuqe.git
```

## Examples
Example of fine-tuning using WMT19 EN-DE QE data on the XLMR-large model.
```
cd $XLMR_PATH
wget https://dl.fbaipublicfiles.com/fairseq/models/xlmr.large.tar.gz
tar -zxvf xlmr.large.tar.gz

export CUDA_VISIBLE_DEVICES=0

python $FAIRSEQ_PATH/fairseq_cli/train.py \
    $NJUQE_PATH/wmt19_ende_data_preprocessed \
    --arch xlmr_qe_large --task qe --criterion qe_base \
    --optimizer adam --clip-norm 1.0 --skip-invalid-size-inputs-valid-test --dataset-impl raw \
    --reset-meters --reset-optimizer --reset-dataloader \
    --src en --mt de --mt-tag tags --score hter --bounds mt --prepend-bos --append-eos \
    --predict-target --predict-score --mask-symbol --fine-tune --qe-meter --joint \
    --ok-loss-weight 1 --score-loss-weight 1 --sent-pooling mixed --best-checkpoint-metric "pearson" --maximize-best-checkpoint-metric \
    --lr 0.000001 --lr-scheduler fixed --max-sentences 1 --max-epoch 50 --patience 10 \
    --update-freq 20 --batch-size-valid 20 --save-interval-updates 300 --validate-interval-updates 300 --no-save \
    --user-dir $NJUQE_PATH/njuqe \
    --restore-file $XLMR_PATH/xlmr-large-model.pt
```

## Citation
Please cite as:
``` bibtex
@inproceedings{geng2023clqe,
  title={Denoising Pre-Training for Machine Translation Quality Estimation with Curriculum Learning},
  author={Geng, Xiang and Zhang, Yu and Li, Jiahuan and Huang, Shujian and Yang, Hao and Tao, Shimin and Chen, Yimeng and Xie, Ning and Chen, Jiajun},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}
```

## Contributor
Xiang Geng (gx@smail.nju.edu.cn), Yu Zhang, Wohao Zhang, Yiming Yan, Qu Cui
