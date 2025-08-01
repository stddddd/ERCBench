#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3

WORK_DIR="/home/jingran/MyBench/lab_AVT/DialogueRNN"
DATASET="iemocap"
echo "${DATASET}"

LOG_PATH="${WORK_DIR}/analysis/${DATASET}"
if [[ ! -d ${LOG_PATH} ]];then
    mkdir -p  ${LOG_PATH}
fi


SEED="0 1 2 3 4"
models="VT"

for model in ${models[@]}
do
for seed in ${SEED[@]}
do
    echo "${model}, ${seed}"
    python -u ${WORK_DIR}/train_ie.py \
    --fea_model ${model} \
    --seed ${seed} \
    >> ${LOG_PATH}/${model}_${seed}.out

done
done