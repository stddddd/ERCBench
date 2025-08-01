#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

WORK_DIR="/data/jingran/MyBench/lab_confi/MultiEMO-confi/eval"
DATASET="meld"
echo "${DATASET}"




SEED="0"
models="AVT"
methods="T-scale"

for method in ${methods[@]}
do
    LOG_PATH="${WORK_DIR}/eval/log-${method}/MELD"
    if [[ ! -d ${LOG_PATH} ]];then
        mkdir -p  ${LOG_PATH}
    fi
for model in ${models[@]}
do
for seed in ${SEED[@]}
do
    echo "${method}, ${model}, ${seed}"
    python Train/TrainMultiEMO.py  --dataset 'MELD' --batch_size 100 --num_layer 2 --num_epochs 50 \
    --SWFC_loss_param 0.3 --HGR_loss_param 0.3 --CE_loss_param 0.4 --sample_weight_param 1.2 \
    --temp_param 1.4 --fea_model ${model} --seed ${seed} --method ${method}  2>&1 | tee ${LOG_PATH}/${model}_${seed}.put    

done
done
done