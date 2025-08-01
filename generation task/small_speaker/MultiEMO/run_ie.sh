#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0


WORK_DIR="/data/jingran/MyBench/lab_speaker/MultiEMO"
DATASET="iemocap"
echo "${DATASET}"

methods="T-scale"
SEED="0"
models="seen unseen"

for method in ${methods[@]}
do
    LOG_PATH="${WORK_DIR}/log-${method}/IEMOCAP"
    if [[ ! -d ${LOG_PATH} ]];then
        mkdir -p  ${LOG_PATH}
    fi
for model in ${models[@]}
do
for seed in ${SEED[@]}
do
    echo "${model}, ${seed}"
    python Train/TrainMultiEMO.py  --dataset 'IEMOCAP' --num_layer 6 --batch_size 64 \
    --SWFC_loss_param 0.4 --HGR_loss_param 0.4 --CE_loss_param 0.2 --sample_weight_param 1.1 \
    --focus_param 2.4  --fea_model ${model} --seed ${seed} --method ${method}  2>&1 | tee ${LOG_PATH}/${model}_${seed}.put    

done
done
done