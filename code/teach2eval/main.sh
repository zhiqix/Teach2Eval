# 输入模型和参数设置
#large_model_name="Yi-1___5-6B-Chat"
#small_model_name="Qwen2.5_1.5B_Instruct"

large_model_name=$1
small_model_name=$2
turn_num=$3

can_tell_answer="True"
file_path="../../results/main_results/${large_model_name}_teach_${small_model_name}"
total_gpu=4

echo "${large_model_name} is teaching ${small_model_name}"

# 执行 Python 脚本并传递参数
python deal_data.py \
  --large_model_name ${large_model_name} \
  --small_model_name ${small_model_name} \
  --can_tell_answer ${can_tell_answer} \
  --file_path ${file_path} \

echo "small_model_name turn 0"

python pipeline.py \
  --model_name ${small_model_name} \
  --can_tell_answer ${can_tell_answer} \
  --file_path ${file_path} \
  --func "dialogue_student" \
  --total_gpu ${total_gpu} \
  --turn 0 \

for turn in $(seq 1 $turn_num)
do
  echo "large_model_name turn ${turn}"
  python pipeline.py \
    --model_name ${large_model_name} \
    --can_tell_answer ${can_tell_answer} \
    --file_path ${file_path} \
    --func "dialogue_teacher" \
    --total_gpu ${total_gpu} \
    --turn ${turn} \

  echo "small_model_name turn ${turn}"
  python pipeline.py \
    --model_name ${small_model_name} \
    --can_tell_answer ${can_tell_answer} \
    --file_path ${file_path} \
    --func "dialogue_student" \
    --total_gpu ${total_gpu} \
    --turn ${turn} \

done

echo "finish testing"

python save_results.py \
  --file_path ${file_path} \
  

