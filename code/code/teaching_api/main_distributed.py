import os
import json
import argparse
import sys
sys.path.append("..")
from src.find_json_files import find_json_files
from pipeline import pipeline

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

def merge_and_split_json_files(input_dir):
    #整理结果的函数
    all_items = []
    new_files = []
    
    # 遍历文件夹中的所有 json 文件
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if os.path.isfile(file_path) and filename.endswith('.json'):
            # 读取每个 JSON 文件
            with open(file_path, 'r') as f:
                data = json.load(f)
                all_items.extend(data)  # 将文件中的列表合并到 all_items 中
            
    # 按照 dataset_name 进行分组并保存
    grouped_data = {}
    for item in all_items:
        dataset_name = item.get('dataset_name')
        if dataset_name:
            del item['dataset_name']
            if dataset_name not in grouped_data:
                grouped_data[dataset_name] = []
            grouped_data[dataset_name].append(item)
    
    # 将分组后的数据保存到对应的 JSON 文件中
    for dataset_name, items in grouped_data.items():
        output_file = os.path.join(input_dir, f"{dataset_name}.json")
        new_files.append(output_file)  # 记录新生成的文件路径
        with open(output_file, 'w') as f:
            json.dump(items, f, indent=2)
        print(f"Saved {len(items)} items to {output_file}")

    # 删除原始的 JSON 文件，排除新生成的文件
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if os.path.isfile(file_path) and filename.endswith('.json') and file_path not in new_files:
            os.remove(file_path)

    return

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
    args = parser.parse_args()
    large_model_name = args.large_model_name
    small_model_name = args.small_model_name
    strategy = args.strategy
    can_tell_answer = args.can_tell_answer
    print(f"{large_model_name} is teaching {small_model_name}")
    
    folder_path = "../../dataset/dataset_lite"
    output_file = f"../../results/teaching/{strategy}:{large_model_name}_to_{small_model_name}"
    json_files = find_json_files(folder_path)

    dataset = []
    for json_file in json_files:
        file_name = os.path.basename(json_file)
        dataset_name = os.path.splitext(file_name)[0]
        
        large_model_results = get_dataset_results(large_model_name, dataset_name)
        small_model_results = get_dataset_results(small_model_name, dataset_name)
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                index = item.get("index",None)
                is_correct = item.get('is_correct',None)
                if is_correct == False:
                    continue
                
                answer_large = large_model_results.get(str(index),None)
                answer_small = small_model_results.get(str(index),None)
                item['result0_small'] = answer_small
                item['result_large'] = answer_large
                item['dataset_name'] = dataset_name
                dataset.append(item)
    print(len(dataset))
    
    #执行batch推理的pipeline
    pipeline(large_model_name, small_model_name, dataset, strategy, can_tell_answer, output_file)
    
    #整理结果
    #merge_and_split_json_files(output_file)
    


