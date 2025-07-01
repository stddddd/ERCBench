#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

WORK_DIR="/home/jingran/MyBench/MMGCN"
DATASET="iemocap"
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
    python -u ${WORK_DIR}/train.py --base-model 'LSTM' --graph-model --nodal-attention \
    --dropout 0.4 --lr 0.0003 --batch-size 16 --l2 0.00003 --graph_type='MMGCN' --epochs=60 \
    --graph_construct='direct' --multi_modal --mm_fusion_mthd='concat_subsequently' --modals='avl' \
    --Dataset='IEMOCAP' --Deep_GCN_nlayers 4 --class-weight --use_speaker \
    --fea_model ${model} \
    --seed ${seed} \
    >> ${LOG_PATH}/${model}_${seed}.out

done
done