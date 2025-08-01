#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3

WORK_DIR="/home/jingran/MyBench/lab_AVT/MultiEMO"
DATASET="iemocap"
echo "${DATASET}"

LOG_PATH="${WORK_DIR}/analysis/IEMOCAP"
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
    python Train/TrainMultiEMO.py  --dataset 'IEMOCAP' --num_layer 6 --batch_size 64 \
    --SWFC_loss_param 0.4 --HGR_loss_param 0.4 --CE_loss_param 0.2 --sample_weight_param 1.1 \
    --focus_param 2.4  --fea_model ${model} --seed ${seed}  2>&1 | tee ${LOG_PATH}/${model}_${seed}.put    

done
done