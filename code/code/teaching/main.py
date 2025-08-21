import os
import json
import re
from dialogue import dialogue
import datetime
import argparse
import time
import multiprocessing
import sys
sys.path.append("..")
from src.find_json_files import find_json_files

def str2bool(value):
    if value.lower() in ('true', '1', 't', 'y', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'f', 'n', 'no'):
        return False
    else:
        raise ValueError("Boolean value expected.")

def get_dataset_results(model_name, dataset_name):
    # 构造文件路径
    file_path = f"../../results/test_model/test_{model_name}.json"
    # 检查文件是否存在
    if not os.path.exists(file_path):
        return {}
    
    # 打开并加载 JSON 文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
    item = data.get(dataset_name,{})
    dataset_results = item.get('dataset_results',{})
    return dataset_results

def test_json(large_model_name, small_model_name, strategy, can_tell_answer, dataset_path, dataset_name, output_path):

    large_model_results = get_dataset_results(large_model_name, dataset_name)
    small_model_results = get_dataset_results(small_model_name, dataset_name)
    print(f"    Testing {dataset_name}")
    results = {}
    
    #建立json文件
    if not os.path.exists(output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({}, f, ensure_ascii=False, indent=4)

    with open(output_path, 'r', encoding='utf-8') as f:
        file_content = f.read().strip()
        # 如果文件内容不为空，加载JSON
        if file_content:
            results = json.loads(file_content)
    # 打开并读取 JSON 文件
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            index = item.get("index",None)

            is_correct = item.get('is_correct',None)
            if str(index) in results:
                continue
            if is_correct == False:
                continue
            
            answer_large = large_model_results.get(str(index),None)
            answer_small = small_model_results.get(str(index),None)
            result = {
                'result0_small': answer_small,
                'result_large': answer_large,
            }
            
            if index % 10 == 0: #todo
                current_time = datetime.datetime.now() + datetime.timedelta(hours=8)
                formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
                #print(f"    index:{index}, current time: {formatted_time}, testing {dataset_name}")
                print(f"        index:{index} — index:{index+9}, current time: {formatted_time}, testing {dataset_name}")
            
            teaching_result = dialogue(
                teacher = large_model_name, 
                student = small_model_name,
                strategy = strategy, 
                can_tell_answer = can_tell_answer,
                item = item,
                dataset_name = dataset_name
                )
            result.update(teaching_result)
            
            results[index] = result
            if index % 1 == 0:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    return

def test_json_wrapper(json_file, large_model_name, small_model_name, strategy, can_tell_answer, output_file):
    """ 包装函数，用于并行处理每个 JSON 文件 """
    file_name = os.path.basename(json_file)
    dataset_name = os.path.splitext(file_name)[0]
    output_path = os.path.join(output_file, file_name)

    try:
        # 调用测试函数
        test_json(
            large_model_name=large_model_name,
            small_model_name=small_model_name,
            strategy=strategy,
            can_tell_answer=can_tell_answer,
            dataset_path=json_file,
            dataset_name=dataset_name,
            output_path=output_path
        )
        print(f"Successfully tested {dataset_name}")
    except Exception as e:
        print(f"Error testing {dataset_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get model_name from command line arguments.")
    parser.add_argument(
        "--large_model_name",
        type=str,
        required=False,
        help="The name of the large model to use.",
        default="Qwen2___5-14B-Instruct",
        )
    parser.add_argument(
        "--small_model_name",
        type=str,
        required=False,
        help="The name of the small model to use.",
        default="Qwen2.5_1.5B_Instruct",
        )
    parser.add_argument(
        "--strategy",
        type=str,
        required=False,
        help="The name of the strategy to use",
        default="base",
        choices=["base", "example", "critique","knowledge","decomposition","socrates","all"]
        )
    parser.add_argument(
        "--can_tell_answer",
        type=str2bool,
        required=False,
        default=False,
        help="whether the teacher can directly tell the student the right answer"
    )
    parser.add_argument(
        "--multithread",
        type=str2bool,
        required=False,
        default=False,
        help="Whether the program is multithreaded"
    )
    args = parser.parse_args()
    large_model_name = args.large_model_name
    small_model_name = args.small_model_name
    strategy = args.strategy
    can_tell_answer = args.can_tell_answer
    multithread = args.multithread
    print(f"{large_model_name} is teaching {small_model_name}")
    
    folder_path = "../../dataset/dataset_lite"
    output_file = f"../../results/teaching/{strategy}:{large_model_name}_to_{small_model_name}"
    if not os.path.exists(output_file):
        os.makedirs(output_file)
    json_files = find_json_files(folder_path)

    if multithread == False:
        for json_file in json_files:
            file_name = os.path.basename(json_file)
            dataset_name = os.path.splitext(file_name)[0]
            output_path = os.path.join(output_file,file_name)
            
            try:
                # 调用测试函数
                test_json(
                    large_model_name = large_model_name,
                    small_model_name = small_model_name,
                    strategy = strategy,
                    can_tell_answer = can_tell_answer,
                    dataset_path = json_file,
                    dataset_name = dataset_name,
                    output_path = output_path,
                    )
                print(f"Successfully tested {dataset_name}")
            except Exception as e:
                print(f"Error testing {dataset_name}: {e}")
    else:
        num_workers = min(multiprocessing.cpu_count(), len(json_files),20)
        with multiprocessing.Pool(processes=num_workers) as pool:
            # 使用 starmap 将参数传递给 process_json_wrapper
            pool.starmap(
                test_json_wrapper,
                [(json_file, large_model_name, small_model_name, strategy, can_tell_answer, output_file) for json_file in json_files]
            )