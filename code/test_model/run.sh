
large_model_names=()
for large_model_name in "${large_model_names[@]}"; do
  nohup python main.py --model_name ${large_model_name} > test_model_${large_model_name}.log 2>&1 &
done