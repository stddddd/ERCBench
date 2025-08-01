#!/bin/bash

# 定义检测 GPU 状态的函数
check_free_gpus() {
  nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits | while read -r line; do
    free=$(echo "$line" | awk -F, '{print $1}') # 获取空闲显存
    total=$(echo "$line" | awk -F, '{print $2}') # 获取总显存
    usage=$(awk "BEGIN {print $free / $total}") # 计算利用率

    # 判断利用率是否大于 80%（空闲）
    if (( $(echo "$usage > 0.1" | bc -l) )); then
      echo "$i"
      return
    fi
    i=$((i+1))
  done
}

# 设置需要运行的命令或脚本
RUN_SCRIPT="/home/jingran/InstructERC-topic/code/train_and_inference_Uni.sh" # 指定脚本路径
RUN_COMMAND="bash $RUN_SCRIPT" # 如果直接运行脚本，使用该命令

# 如果直接写命令，而不是依赖脚本，可以改成如下内容：
# RUN_COMMAND=\"python run_watermarking_gsm8k.py \\
#     --model_name \\$model_path \\
#     ... (省略其他命令参数，保持一致)\"


# 循环检测 GPU 状态
while true; do
  free_gpu=$(check_free_gpus)

  if [ -n "$free_gpu" ]; then
    echo "Free GPU detected: $free_gpu"
    
    # 设置指定 GPU 并运行程序
    export CUDA_VISIBLE_DEVICES=$free_gpu
    echo "Running command: $RUN_COMMAND"
    eval $RUN_COMMAND
    exit 0
  else
    echo "No free GPUs detected. Retrying in 100 seconds..."
    sleep 100
  fi
done