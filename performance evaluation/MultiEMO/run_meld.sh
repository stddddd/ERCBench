#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2

WORK_DIR="/home/jingran/MyBench/lab_AVT/MultiEMO"
DATASET="meld"
echo "${DATASET}"

LOG_PATH="${WORK_DIR}/analysis/MELD"
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
    python Train/TrainMultiEMO.py  --dataset 'MELD' --batch_size 100 --num_layer 2 --num_epochs 15 \
    --SWFC_loss_param 0.3 --HGR_loss_param 0.3 --CE_loss_param 0.4 --sample_weight_param 1.2 \
    --temp_param 1.4 --fea_model ${model} --seed ${seed}  2>&1 | tee ${LOG_PATH}/${model}_${seed}.put    

done
done