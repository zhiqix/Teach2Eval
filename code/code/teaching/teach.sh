#!/bin/bash

# 定义循环的参数（例如不同的策略或模型名称）
large_model_names=("Llama-3___3-70B-Instruct" "Meta-Llama-3-8B-Instruct" "Meta-Llama-3___1-8B-Instruct" "Meta-Llama-3___1-70B-Instruct" "internlm2_5-7b" "internlm2_5-20b")
small_model_names=("MiniCPM-2B-dpo-bf16" "Llama-3___2-1B-Instruct")

# 循环遍历参数组合

for large_model_name in "${large_model_names[@]}"; do
  for small_model_name in "${small_model_names[@]}"; do
    # 调用deal.sh脚本并传递参数
    bash run.sh  "$large_model_name" "$small_model_name" > output_${large_model_name}_teach_${small_model_name}.log 2>&1
  done
done

echo "Finished running all combinations"
