import os
import json
import re
import datetime
import argparse
from pipeline import pipeline

def find_json_files(folder_path):
    json_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def deal_results(data, output_path):
    result_dict = {}

    for item in data:
        dataset_name = item["dataset_name"]
        index = item["index"]
        result = item["result"]

        # 如果 dataset_name 已经在字典中，添加到已有的列表中
        if dataset_name in result_dict:
            result_dict[dataset_name]["dataset_results"][index] = result
        else:
            # 如果 dataset_name 不在字典中，创建新的列表
            result_dict[dataset_name] = {"dataset_results":{index:result}}
            

    for value in result_dict.values():
        dataset_results = value['dataset_results']
        total_count = 0
        correct_count = 0
        for result in dataset_results.values():
            total_count += 1
            if result[0] == 1:
                correct_count += 1
        value['correct_count'] = correct_count
        value['accuracy_rate'] = correct_count/total_count
        
    with open(output_path, 'w') as f:
        json.dump(result_dict, f, indent=4)
    
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get model_name from command line arguments.")
    parser.add_argument("--model_name", type=str, required=True, help="The name of the model to use.")
    args = parser.parse_args()
    model_name = args.model_name
    total_gpu = 4
    
    print(f"Testing model: {model_name}")
    
    
    folder_path = "../../dataset/dataset_test"
    output_file = "../../results/test_model"
    output_path = os.path.join(output_file, 'test_' + model_name + '.json')
    json_files = find_json_files(folder_path)
            
    dataset = []
    with open("chosen_data_v2.json",'r', encoding='utf-8') as file:
        chosen_data_dict = json.load(file)
    for json_file in json_files:
        file_name = os.path.basename(json_file)
        dataset_name = os.path.splitext(file_name)[0]
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                is_correct = item.get('is_correct',None)
                if is_correct == False:
                    continue
                index = item['index']
                if dataset_name in chosen_data_dict and str(index) in chosen_data_dict[dataset_name]:
                    item['dataset_name'] = dataset_name
                    dataset.append(item)
    results = pipeline(model_name, dataset, total_gpu)

    results_dict = deal_results(results, output_path)
