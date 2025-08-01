#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3

WORK_DIR="/data/jingran/MyBench/lab_topic/DialogueRNN"
DATASET="meld"
echo "${DATASET}"

LOG_PATH="${WORK_DIR}/log3/${DATASET}"
if [[ ! -d ${LOG_PATH} ]];then
    mkdir -p  ${LOG_PATH}
fi


SEED="0 1 2 3 4"
models="seen unseen"

for model in ${models[@]}
do
for seed in ${SEED[@]}
do
    echo "${model}, ${seed}"
    python -u ${WORK_DIR}/train_me.py \
    --fea_model ${model} \
    --seed ${seed} \
    >> ${LOG_PATH}/${model}_${seed}.out

done
done