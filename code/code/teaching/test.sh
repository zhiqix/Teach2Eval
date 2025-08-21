# 输入模型和参数设置
large_model_name="Qwen2___5-14B-Instruct"
small_model_name="Qwen2.5_1.5B_Instruct"

#large_model_name=$1
#small_model_name=$2

strategy="base"
can_tell_answer="False"
file_path="../../results/main_results/${strategy}_${large_model_name}_teach_${small_model_name}"
total_gpu=4

echo "${large_model_name} is teaching ${small_model_name}"

# 执行 Python 脚本并传递参数
python deal_data.py \
  --large_model_name ${large_model_name} \
  --small_model_name ${small_model_name} \
  --strategy ${strategy} \
  --can_tell_answer ${can_tell_answer} \
  --file_path ${file_path} \