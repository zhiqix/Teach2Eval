#!/bin/bash

# 定义循环的参数（例如不同的策略或模型名称）
large_model_names=("gemini-2.5-flash")
small_model_names=("internlm2_5-1_8b-chat")
# small_model_names=("Qwen2.5_1.5B_Instruct" "Llama-3___2-1B-Instruct" "MiniCPM-2B-dpo-bf16" "internlm2_5-1_8b-chat")
turn_num=3

# 循环遍历参数组合

for large_model_name in "${large_model_names[@]}"; do
  for small_model_name in "${small_model_names[@]}"; do
    # 调用deal.sh脚本并传递参数
    bash main.sh  "$large_model_name" "$small_model_name" "$turn_num" > output_${large_model_name}_teach_${small_model_name}.log 2>&1
  done
done

echo "Finished running all combinations"
