import os
import json
import argparse


def split_json_files(all_items, input_dir):            
    # 按照 dataset_name 进行分组并保存
    grouped_data = {}
    for item in all_items:
        dataset_name = item.get('dataset_name')
        if dataset_name:
            del item['dataset_name']
            del item['whole_question']
            del item['whole_question_wo_options']
            if dataset_name not in grouped_data:
                grouped_data[dataset_name] = []
            grouped_data[dataset_name].append(item)
    
    # 将分组后的数据保存到对应的 JSON 文件中
    for dataset_name, items in grouped_data.items():
        output_file = os.path.join(input_dir, f"{dataset_name}.json")
        with open(output_file, 'w') as f:
            json.dump(items, f, indent=2)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get model_name from command line arguments.")
    parser.add_argument(
        "--file_path",
        type=str,
        required=False,
        default="",
    )
    args = parser.parse_args()
    file_path = args.file_path
    file = os.path.join(file_path, "data.json")
    with open(file, 'r', encoding='utf-8') as f:
        all_items = json.load(f)
        
    split_json_files(all_items, file_path)

    os.remove(file)
    