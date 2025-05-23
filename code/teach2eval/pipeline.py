import argparse
import os
import json

import concurrent.futures
from dialogue_student import dialogue_student
from dialogue_teacher import dialogue_teacher
import sys
sys.path.append("..")
from model import model_gpu_use, model_use_api
from utils.function import str2bool

def split_numbers(total_gpu, parallel_size):    
    if parallel_size > total_gpu:
        return [str(i) for i in range(total_gpu + 1)]
    base_size = total_gpu // parallel_size
    remainder = total_gpu % parallel_size
    result = []
    start = 0
    for i in range(parallel_size):
        if i < remainder:
            end = start + base_size + 1
        else:
            end = start + base_size
        group_str = ",".join(str(num) for num in range(start, end))
        result.append(group_str)
        start = end
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="The name of the model to use.",
        )
    parser.add_argument(
        "--can_tell_answer",
        type=str2bool,
        required=False,
        default=False,
        help="whether the teacher can directly tell the student the right answer"
        )
    parser.add_argument(
        "--file_path",
        type=str,
        required=False,
        default="",
        )
    parser.add_argument(
        "--func",
        type=str,
        required=True,
        )
    parser.add_argument(
        "--total_gpu",
        type=int,
        required=False,
        default=4
        )
    parser.add_argument(
        "--turn",
        type=int,
        required=True
        )
    args = parser.parse_args()
    model_name = args.model_name
    can_tell_answer = args.can_tell_answer
    file_path = args.file_path
    func = args.func
    total_gpu = args.total_gpu
    turn = args.turn
    use_api = model_use_api[model_name]
    
    if use_api == True:
        parallel_size = 64
        gpu_id_list = range(parallel_size)
        max_workers=parallel_size
    else:
        parallel_size = total_gpu // model_gpu_use[model_name]
        gpu_id_list = split_numbers(total_gpu, parallel_size)
        max_workers=total_gpu

    file = os.path.join(file_path, "data.json")
    with open(file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    split_dataset = []
    batch_size = (len(dataset) - 1) // parallel_size + 1
    for batch_idx in range(parallel_size):
        start_index = batch_idx * batch_size
        end_index = min((batch_idx + 1) * batch_size, len(dataset))
        split_dataset.append(dataset[start_index: end_index])
    
    all_results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务并返回 Future 对象
        futures = [
            executor.submit(globals()[func], id, split_dataset[id], gpu_id_list[id], model_name, can_tell_answer, turn)
            for id in range(parallel_size)
        ]
        
        # 获取每个任务的结果，按任务完成顺序
        for future in concurrent.futures.as_completed(futures):
            all_results.extend(future.result())

    with open(file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
        
