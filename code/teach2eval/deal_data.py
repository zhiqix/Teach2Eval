import os
import json
import argparse
import sys
import random
sys.path.append("..")
from utils.find_json_files import find_json_files
from utils.function import str2bool

def get_dataset_results(model_name, dataset_name):
    # 构造文件路径
    file_path = f"../../results/test_model/test_{model_name}.json"
    # 打开并加载 JSON 文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    item = data.get(dataset_name,{})
    dataset_results = item.get('dataset_results',{})
    return dataset_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        "--file_path",
        type=str,
        required=False,
        default="",
        )
    parser.add_argument(
        "--dataset_chosen",
        type=str,
        required=False,
        default="all"
        )
    args = parser.parse_args()
    large_model_name = args.large_model_name
    small_model_name = args.small_model_name
    strategy = args.strategy
    can_tell_answer = args.can_tell_answer
    file_path = args.file_path 
    dataset_chosen = args.dataset_chosen

    input_path = "../../dataset/dataset_test"
    json_files = find_json_files(input_path)

    dataset = []

    for json_file in json_files:
        file_name = os.path.basename(json_file)
        dataset_name = os.path.splitext(file_name)[0]
        
        if dataset_chosen != "all" and dataset_chosen != dataset_name:
            continue
        
        large_model_results = get_dataset_results(large_model_name, dataset_name)
        #small_model_results = get_dataset_results(small_model_name, dataset_name)

        with open(json_file, 'r', encoding='utf-8') as f:
            items = json.load(f)
            for item in items:
                
                is_correct = item.get('is_correct',False)
                if is_correct == False:
                    continue
            
                index = item.get('index',-1)
                question = item.get('question',None)
                options = item.get('options',[])
                passage = item.get('passage',None)
                label = item.get('label',[])
                question_type = item.get('type',-1)
                if not passage == None:
                    question = passage + '\n' + question
                options_str = "Options:\n   " + "\n   ".join(options)
                whole_question = "Question: " + question + "\n" + options_str
                whole_question_wo_options = "Question: " + question
                answer_large = large_model_results.get(str(index),None)
                #answer_small = small_model_results.get(str(index),None)

                data = {
                    "dataset_name": dataset_name,
                    "index": index,
                    "question": question,
                    "options": options,
                    "label": label,
                    "question_type": question_type,
                    "strategy": strategy,
                    "can_tell_answer": can_tell_answer,
                    "result_large": answer_large,
                    "whole_question": whole_question,
                    "whole_question_wo_options": whole_question_wo_options,
                    "conversation": [],
                }
                
                item['dataset_name'] = dataset_name
                dataset.append(data)        
                
#dataset = random.sample(dataset, 30)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    output_file = os.path.join(file_path, "data.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)
        

