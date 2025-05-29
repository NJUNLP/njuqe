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

Example of generating pseudo translations with constrained beam search.
```
python $FAIRSEQ_PATH/fairseq_cli/generate.py \
    $PARALLEL_BPE_DATA_PATH \
    --path $TRANSLATION_MODEL_PATH \
    --dataset-impl raw --gen-subset $SUBSETNAME --skip-invalid-size-inputs-valid-test --remove-bpe \
    --task cbs_translation --beam 5 --batch-size 512 \
    --threshold-prob 0.1 --lamda-ratio 0.55 --softmax-temperature 0.20 \
    --user-dir $NJUQE_PATH/njuqe
```

Download tercom from https://www.cs.umd.edu/~snover/tercom/, then use $NJUQE_PATH/scripts/ter/generate_ter_label.sh 
generate pseudo labels. Or design other labeling rules for specific annotations.

Example of generating pseudo MQM QE data with DCSQE.
```
bash scripts/generate_dcsqe.sh
```
You have to prepare the parallel data and the Generator and the Annotator before run above script.
```
$SEED_DATA_PATH/                   # Parallel data
├── raw.$src-$tgt.$src                  
├── raw.$src-$tgt.$tgt
├── dict.$src.txt                  # The dictionaries corresponding to Generator and Annotator
├── dict.$tgt.txt
├── dict.tag.txt                   # including "OK 3\nBAD 2\nPAD 1"
├── bpecode.${src}-${tgt}.joint
```




## Citation
Please cite as:
``` bibtex
@inproceedings{geng2025dcsqe,
  title={Alleviating Distribution Shift in Synthetic Data for Machine Translation Quality Estimation}, 
  author={Xiang Geng and Zhejian Lai and Jiajun Chen and Hao Yang and Shujian Huang},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics},
  year={2025}
}

@inproceedings{geng2023cbsqe,
  title={Improved Pseudo Data for Machine Translation Quality Estimation with Constrained Beam Search},
  author={Geng, Xiang and Zhang, Yu and Lai, Zhejian and She, Shuaijie and Zou, Wei and Tao, Shimin and Yang, Hao and Chen, Jiajun and Huang, Shujian},
  booktitle={Conference on Empirical Methods in Natural Language Processing},
  year={2023}
}

@inproceedings{geng2023clqe,
  title={Denoising Pre-Training for Machine Translation Quality Estimation with Curriculum Learning},
  author={Geng, Xiang and Zhang, Yu and Li, Jiahuan and Huang, Shujian and Yang, Hao and Tao, Shimin and Chen, Yimeng and Xie, Ning and Chen, Jiajun},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}

@inproceedings{cui2021directqe,
  title={Directqe: Direct pretraining for machine translation quality estimation},
  author={Cui, Qu and Huang, Shujian and Li, Jiahuan and Geng, Xiang and Zheng, Zaixiang and Huang, Guoping and Chen, Jiajun},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={14},
  pages={12719--12727},
  year={2021}
}
```

## Contributor
Xiang Geng (gx@smail.nju.edu.cn), Yu Zhang, Zhejian Lai(laizj@smail.nju.edu.cn), Wohao Zhang, Yiming Yan, Qu Cui
