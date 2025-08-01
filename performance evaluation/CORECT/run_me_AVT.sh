#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3

WORK_DIR="/home/jingran/MyBench/lab_AVT/CORECT"
DATASET="meld"
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
    # echo "${model}, ${seed}, data"
    # python -u ${WORK_DIR}/preprocess.py --dataset="${DATASET}" \
    # --fea_model ${model} --seed ${seed} >> ${LOG_PATH}/${model}_${seed}.out
    echo "${model}, ${seed}, train"
    python -u ${WORK_DIR}/train.py --dataset="${DATASET}" --modalities="atv" --from_begin \
    --epochs=50 --learning_rate=0.00025 --optimizer="adam" --drop_rate=0.5 --batch_size=10 \
    --rnn="transformer" --use_speaker  --edge_type="temp_multi" --wp=11 --wf=5  --gcn_conv="rgcn" \
    --use_graph_transformer --graph_transformer_nheads=7  --use_crossmodal --num_crossmodal=2 \
    --num_self_att=3 --crossmodal_nheads=2 --self_att_nheads=2 \
    --fea_model ${model} --seed ${seed} >> ${LOG_PATH}/${model}_${seed}.out
    # python eval.py --dataset="${DATASET}" --modalities="atv" >> ${LOG_PATH}/${model}_${seed}.out

done
done