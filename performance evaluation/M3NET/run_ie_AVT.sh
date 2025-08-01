#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2

WORK_DIR="/home/jingran/MyBench/lab_AVT/M3NET"
DATASET="iemocap"
echo "${DATASET}"

LOG_PATH="${WORK_DIR}/analysis/${DATASET}"
if [[ ! -d ${LOG_PATH} ]];then
    mkdir -p  ${LOG_PATH}
fi


SEED="0 1 2 3 4"
models="AVT"

for model in ${models[@]}
do
for seed in ${SEED[@]}
do
    echo "${model}, ${seed}"
    python -u ${WORK_DIR}/train.py --base-model 'GRU' \
    --dropout 0.5 --lr 0.0001 --batch-size 16 --graph_type='hyper' --epochs=80 \
    --graph_construct='direct' --multi_modal --mm_fusion_mthd='concat_DHT' \
    --modals='avl' --Dataset='IEMOCAP' --norm BN --num_L=3 --num_K=4 \
    --fea_model ${model} \
    --seed ${seed} \
    >> ${LOG_PATH}/${model}_${seed}.out

done
done