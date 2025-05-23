from get_answer import get_answer
import concurrent.futures
import sys
sys.path.append("..")
from model import model_use_api,model_gpu_use

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

def pipeline(model_name, dataset, total_gpu):
    split_dataset = []
    all_results = []
        
    batch_size = (len(dataset) - 1) // parallel_size + 1
    for batch_idx in range(parallel_size):
        start_index = batch_idx * batch_size
        end_index = min((batch_idx + 1) * batch_size, len(dataset))
        split_dataset.append(dataset[start_index: end_index])

    use_api = model_use_api[model_name]
    if use_api == True:
        parallel_size = 64
        gpu_id_list = range(parallel_size)
        max_workers=parallel_size
    else:
        parallel_size = total_gpu // model_gpu_use[model_name]
        gpu_id_list = split_numbers(total_gpu, parallel_size)
        max_workers=total_gpu
        
    gpu_id_list = split_numbers(total_gpu, parallel_size)
    
    # 使用 concurrent.futures.ProcessPoolExecutor 来进行并行处理
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = [
            executor.submit(get_answer, split_dataset[id], gpu_id_list[id], model_name)
            for id in range(parallel_size)
        ]
        
        # 获取所有任务的结果
        for future in concurrent.futures.as_completed(futures):
            result = future.result()  # 获取任务的返回值
            all_results.extend(result)
    
    return all_results
