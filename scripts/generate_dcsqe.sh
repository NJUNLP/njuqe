#!/usr/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
BPE_ROOT=
TER_ROOT=
FAIRSEQ_PATH=
SEED_DATA_PATH=
GENRATOR_PATH=
ANNOTATOR_PATH=

seed=26
src=en
tgt=de
num=500000
prob=0.05
lamda=0.2
type=dcsqe_lambda${lamda}_prob${prob}

output_dir=${src}-${tgt}/$((num / 10000))w/${type}
mkdir -p $output_dir

cp ${BASH_SOURCE[0]} $output_dir/generate.sh
# cp ${0}/default_hyperparameters.yaml $output_dir/default_hyperparameters.yaml

# cuda=1
# echo "Using GPU $cuda..."

cp $SEED_DATA_PATH/dict.${src}.txt $output_dir/dict.${src}.txt
cp $SEED_DATA_PATH/dict.${tgt}.txt $output_dir/dict.${tgt}.txt

# 需要创建一个dict.tag.txt到output_dir

python $FAIRSEQ_PATH/fairseq_cli/generate.py \
    ${SEED_DATA_PATH} \
    --task cbs_translation -s ${src} -t ${tgt} --log-format simple --beam 5 \
    --path $GENRATOR_PATH \
    --threshold-prob ${prob} --lamda-ratio ${lamda} --softmax-temperature 0.20  --remove-bpe \
    --max-tokens 4096 --gen-subset train --dataset-impl raw --seed ${seed} --skip-invalid-size-inputs-valid-test \
    --user-dir $FAIRSEQ_PATH/njuqe >${output_dir}/generate.log 2>&1

cat ${output_dir}/generate.log | grep -P "^H" | sort -V | cut -f3 > $output_dir/generate.txt
python ${BPE_ROOT}/apply_bpe.py --num-workers 64 -c $SEED_DATA_PATH/bpecode.${src}-${tgt}.joint <  $output_dir/generate.txt > $output_dir/generate.bpe

python $TER_ROOT/standard-file.py \
    $output_dir/generate.txt \
    $output_dir/generate_standard.txt

python $TER_ROOT/standard-file.py \
    ${SEED_DATA_PATH}/raw.${src}-${tgt}.${tgt} \
    $output_dir/reference_standard.txt

java -jar $TER_ROOT/tercom.7.25.jar \
    -o pra \
    -r $output_dir/reference_standard.txt \
    -h $output_dir/generate_standard.txt \
    -n $output_dir/out

python $TER_ROOT/tercom-transfer.py \
    $output_dir/out.pra \
    $output_dir/generate.tag \
    /dev/null

python get_tag_from_bpe_word.py --raw-hyp $output_dir/generate.txt --bpe-hyp $output_dir/generate.bpe --raw-tag $output_dir/generate.tag --bpe-tag $output_dir/generate.bpe.tag && echo "Tags ha been converted to bpe tags"

mv $output_dir/generate.bpe $output_dir/generate.${src}-${tgt}.${tgt}
mv $output_dir/generate.bpe.tag $output_dir/generate.${src}-${tgt}.tag
cp /home/nfs02/laizj/experiment/WMT2023/generate/pseudo_data/${src}-${tgt}/$((num / 10000))w/raw/train.${src}-${tgt}.${src} $output_dir/generate.${src}-${tgt}.${src}

# head -n 100 $output_dir/generate.${src}-${tgt}.${src} > $output_dir/test.${src}-${tgt}.${src}
# head -n 100 $output_dir/generate.${src}-${tgt}.${tgt} > $output_dir/test.${src}-${tgt}.${tgt}
# head -n 100 $output_dir/generate.${src}-${tgt}.tag > $output_dir/test.${src}-${tgt}.tag

python generate_tags_from_cbsqe.py \
    ${output_dir} \
    --task cbsqe_gen_prob -s ${src} -t ${tgt} \
    --user-dir $FAIRSEQ_PATH/njuqe \
    --max-tokens 4096 --gen-subset generate --seed ${seed} \
    --path $ANNOTATOR_PATH \
    --dataset-impl raw --results-path ${output_dir} \
    --model-overrides "{'eval_bleu': False, 'required_seq_len_multiple':1, 'load_alignments': False}" >${output_dir}/generate.log 2>&1

BIN=${output_dir}/bin
mkdir -p $BIN

cat ${output_dir}/generate${seed}.hyp | grep -P "^H" | sort -V > $output_dir/generate${seed}.sort.hyp
cat $output_dir/generate${seed}.sort.hyp | cut -f2 > $BIN/train.raw.mqm_score
cat $output_dir/generate${seed}.sort.hyp | cut -f3 > $BIN/train.raw.tag
cat $output_dir/generate${seed}.sort.hyp | cut -f4 > $BIN/train.raw.dtag
cp /home/nfs02/laizj/experiment/WMT2023/generate/pseudo_data/${src}-${tgt}/$((num / 10000))w/raw/raw.${src}-${tgt}.${src} $BIN/train.raw.${src}-${tgt}.${src}
cp $output_dir/generate.txt $BIN/train.raw.${src}-${tgt}.${tgt}
sed -i 's/$/ <EOS>/' $BIN/train.raw.${src}-${tgt}.${tgt}

ok_count=$(grep -o OK $BIN/train.raw.tag | wc -l)
bad_count=$(grep -o BAD $BIN/train.raw.tag | wc -l)
bad_weight=2
# 计算OK/BAD
if [ $bad_count -ne 0 ]; then
    ok_bad_ratio=$(echo "scale=2; $ok_count / $bad_count" | bc)
else
    ok_bad_ratio="inf"
fi
# 计算bad_weight / (OK/BAD)
if [ "$ok_bad_ratio" != "inf" ]; then
    ratio=$(echo "scale=2; $bad_weight / $ok_bad_ratio" | bc)
else
    ratio="0 (Cannot divide by infinity)"
fi
echo -e "\nOK: ${ok_count} BAD: ${bad_count} OK/BAD: ${ok_bad_ratio} weight=${ratio}" >>${output_dir}/generate.log 2>&1
echo -e "\n skip number: $(wc -l ${output_dir}/generate${seed}.skip) \n" >>${output_dir}/generate.log 2>&1

python generate_valid.py --input-dir ${BIN} --output-dir ${BIN} --src ${src} --mt ${tgt}
# rm -rf ${BIN}/train.raw.* ${output_dir}/generate_standard.txt ${output_dir}/reference_standard.txt ${output_dir}/out.pra ${output_dir}/generate.txt ${output_dir}/generate.tag ${output_dir}/generate$.${src}-${tgt}.*

python generate_sentencepiece_data.py --input-dir ${BIN} --output-dir ${BIN} --src ${src} --mt ${tgt} --train --valid

for item in ${src} ${tgt} tag dtag; do
    cp "$SEED_DATA_PATH/dict.${item}.txt" "${BIN}/dict.${item}.txt"
done

fairseq-preprocess \
    --source-lang ${src} \
    --target-lang ${tgt} \
    --trainpref ${BIN}/train.${src}-${tgt}.spm \
    --validpref ${BIN}/valid.${src}-${tgt}.spm \
    --destdir ${BIN} \
    --srcdict ${BIN}/dict.${src}.txt \
    --tgtdict ${BIN}/dict.${tgt}.txt \
    --dataset-impl mmap \
    --workers 64

fairseq-preprocess \
    --source-lang tag \
    --srcdict $BIN/dict.tag.txt \
    --trainpref $BIN/train \
    --validpref $BIN/valid \
    --destdir $BIN \
    --only-source \
    --dataset-impl mmap \
    --workers 64
