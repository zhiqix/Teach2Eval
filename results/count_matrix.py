import json
import os
import csv
import pickle
import numpy as np
import argparse

knowledge = ['jecQA', 'medical_meadow_mmmlu', 'medical_meadow_medqa', 'fomc', 'financial_phrasebank', 'MMLU', 'OpenBookQA', 'ARC-E', 'MMLU-Pro', 'GPQA', 'web_questions', 'ARC-C']
multilanguage = ['AGIEval', 'mgsm', 'ceval']
understanding = ['rte', 'BigBenchLite', 'civil_comments', 'RACE', 'semeval-2014', 'QuALITY', 'imdb', 'SST2']
reasoning = ['MATH','web_of_lies', 'formal_fallacies', 'reasoning_about_colored_objects', 'navigate', 'logical_deduction', 'object_counting', 'penguins_in_a_table', 'boolean_expressions', 'aqua', 'gsm8k', 'MultiArith', 'singleq', 'svamp', 'tabmwp', 'vitaminc_fact_verification', 'emoji_movie', 'gre_reading_comprehension', 'winowhy', 'causal_judgement', 'fantasy_reasoning', 'minute_mysteries_qa', 'goal_step_wikihow', 'sports_understanding', 'disambiguation_qa', 'temporal_sequences', 'causality_mcq', 'typical_time_mcq', 'ambiguity_resolution_mcq', 'relation_mcq', 'duration_mcq', 'nli_mcq', 'frequency_mcq', 'ordering_mcq', 'date_understanding', 'arithmetic_mcq', 'storytelling_mcq']

def calculate_percentage(json_file, turn):
    # 读取 JSON 文件
    with open(json_file, 'r') as file:
        data = json.load(file)

    # 初始化计数器
    total_counts = {f"result{i}_small": 0 for i in range(turn)}
    count_ones = {f"result{i}_small": 0 for i in range(turn)}

    total_counts_large = 0
    count_ones_large = 0
    
    # 遍历每条记录
    for record_data in data:
        key = "result_large"
        if key in record_data:
            total_counts_large += 1
            if record_data[key][0] == 1:
                count_ones_large += 1
        # 遍历 result0_small 到 result5_small
        for i in range(turn):
            key = f"result{i}_small"
            if key in record_data:
                total_counts[key] += 1
                if record_data[key][0] == 1:
                    count_ones[key] += 1

    # 计算百分比
    percentages = [
        count_ones[f"result{i}_small"] / total_counts[f"result{i}_small"] if total_counts[f"result{i}_small"] > 0 else 0
        for i in range(turn)
    ]

    percentages.append(count_ones_large / total_counts_large )
    return percentages, total_counts, count_ones, total_counts_large, count_ones_large

def process_folder(folder_path,turn = 3):

    turn += 1
    results = []
    name = os.path.basename(folder_path)
    # 需要改变
    output_csv = f"main_results_v2/{name}/summary.csv"  # 输出的 CSV 文件路径
    total_counts_all = {f"result{i}_small": 0 for i in range(turn)}
    count_ones_all = {f"result{i}_small": 0 for i in range(turn)}
    total_counts_large_all = 0
    count_ones_large_all = 0
    
    total_counts_understanding = {f"result{i}_small": 0 for i in range(turn)}
    count_ones_understanding = {f"result{i}_small": 0 for i in range(turn)}
    total_counts_large_understanding = 0
    count_ones_large_understanding = 0
    
    total_counts_reasoning = {f"result{i}_small": 0 for i in range(turn)}
    count_ones_reasoning = {f"result{i}_small": 0 for i in range(turn)}
    total_counts_large_reasoning = 0
    count_ones_large_reasoning = 0
    
    total_counts_multilanguage = {f"result{i}_small": 0 for i in range(turn)}
    count_ones_multilanguage  = {f"result{i}_small": 0 for i in range(turn)}
    total_counts_large_multilanguage  = 0
    count_ones_large_multilanguage  = 0
    
    total_counts_knowledge = {f"result{i}_small": 0 for i in range(turn)}
    count_ones_knowledge = {f"result{i}_small": 0 for i in range(turn)}
    total_counts_large_knowledge = 0
    count_ones_large_knowledge = 0
    
    # 遍历文件夹中的 JSON 文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            json_file_path = os.path.join(folder_path, filename)
            json_filename = os.path.splitext(os.path.basename(json_file_path))[0]
            percentages, total_counts, count_ones, total_counts_large, count_ones_large = calculate_percentage(json_file_path, turn)
            
            total_counts_large_all += total_counts_large
            count_ones_large_all += count_ones_large
            for i in range(turn):
                total_counts_all[f'result{i}_small'] += total_counts[f'result{i}_small']
                count_ones_all[f'result{i}_small'] += count_ones[f'result{i}_small']
            
            if json_filename in knowledge:
                total_counts_large_knowledge += total_counts_large
                count_ones_large_knowledge += count_ones_large
                for i in range(turn):
                    total_counts_knowledge[f'result{i}_small'] += total_counts[f'result{i}_small']
                    count_ones_knowledge[f'result{i}_small'] += count_ones[f'result{i}_small']
            elif json_filename in multilanguage:
                total_counts_large_multilanguage += total_counts_large
                count_ones_large_multilanguage += count_ones_large
                for i in range(turn):
                    total_counts_multilanguage[f'result{i}_small'] += total_counts[f'result{i}_small']
                    count_ones_multilanguage[f'result{i}_small'] += count_ones[f'result{i}_small']
            elif json_filename in understanding:
                total_counts_large_understanding += total_counts_large
                count_ones_large_understanding += count_ones_large
                for i in range(turn):
                    total_counts_understanding[f'result{i}_small'] += total_counts[f'result{i}_small']
                    count_ones_understanding[f'result{i}_small'] += count_ones[f'result{i}_small']
            elif json_filename in reasoning:
                total_counts_large_reasoning += total_counts_large
                count_ones_large_reasoning += count_ones_large
                for i in range(turn):
                    total_counts_reasoning[f'result{i}_small'] += total_counts[f'result{i}_small']
                    count_ones_reasoning[f'result{i}_small'] += count_ones[f'result{i}_small']
                
            results.append([json_filename] + percentages)
    
    percentages_all = [
        count_ones_all[f"result{i}_small"] / total_counts_all[f"result{i}_small"] if total_counts_all[f"result{i}_small"] > 0 else 0
        for i in range(turn)
    ]
    percentages_all.append(count_ones_large_all / total_counts_large_all)
    results.append(['all'] + percentages_all)
    
    percentages_knowledge = [
        count_ones_knowledge[f"result{i}_small"] / total_counts_knowledge[f"result{i}_small"] if total_counts_knowledge[f"result{i}_small"] > 0 else 0
        for i in range(turn)
    ]
    percentages_knowledge.append(count_ones_large_knowledge / total_counts_large_knowledge * 100)
    results.append(['knowledge'] + percentages_knowledge)
    
    percentages_multilanguage = [
        count_ones_multilanguage[f"result{i}_small"] / total_counts_multilanguage[f"result{i}_small"] if total_counts_multilanguage[f"result{i}_small"] > 0 else 0
        for i in range(turn)
    ]
    percentages_multilanguage.append(count_ones_large_multilanguage / total_counts_large_multilanguage)
    results.append(['multilanguage'] + percentages_multilanguage)
    
    percentages_understanding = [
        count_ones_understanding[f"result{i}_small"] / total_counts_understanding[f"result{i}_small"] if total_counts_understanding[f"result{i}_small"] > 0 else 0
        for i in range(turn)
    ]
    percentages_understanding.append(count_ones_large_understanding / total_counts_large_understanding)
    results.append(['understanding'] + percentages_understanding)
    
    percentages_reasoning = [
        count_ones_reasoning[f"result{i}_small"] / total_counts_reasoning[f"result{i}_small"] if total_counts_reasoning[f"result{i}_small"] > 0 else 0
        for i in range(turn)
    ]
    percentages_reasoning.append(count_ones_large_reasoning / total_counts_large_reasoning)
    results.append(['reasoning'] + percentages_reasoning)
    
    
    # 写入 CSV 文件
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        # 写入表头
        header = ["Filename"] + [f"result{i}" for i in range(turn)] + ["result_large"]
        csv_writer.writerow(header)
        # 写入数据
        csv_writer.writerows(results)

def calculate_matrix(json_file, domain, turn):
    # 读取 JSON 文件
    with open(json_file, 'r') as file:
        data = json.load(file)
    json_filename = os.path.splitext(os.path.basename(json_file))[0]
    matrix1 = np.zeros((turn+1, 2, 2))
    matrix2 = np.zeros((turn, 2, 2))
    if json_filename not in domain:
        return matrix1, matrix2
    for item in data:
        for i in range(turn + 1):
            if item['result_large'][0] == 1 and item[f'result{i}_small'][0] == 1:
                matrix1[i][0][0] += 1
            if item['result_large'][0] == 0 and item[f'result{i}_small'][0] == 1:
                matrix1[i][1][0] += 1
            if item['result_large'][0] == 0 and item[f'result{i}_small'][0] == 0:
                matrix1[i][1][1] += 1
            if item['result_large'][0] == 1 and item[f'result{i}_small'][0] == 0:
                matrix1[i][0][1] += 1
        for i in range(turn):
            if item['result0_small'][0] == 1 and item[f'result{i+1}_small'][0] == 1:
                matrix2[i][0][0] += 1
            if item['result0_small'][0] == 0 and item[f'result{i+1}_small'][0] == 1:
                matrix2[i][1][0] += 1
            if item['result0_small'][0] == 0 and item[f'result{i+1}_small'][0] == 0:
                matrix2[i][1][1] += 1
            if item['result0_small'][0] == 1 and item[f'result{i+1}_small'][0] == 0:
                matrix2[i][0][1] += 1
    return matrix1, matrix2
    
def generate_confushion_matrix(folder_path, turn = 3):
    lists = {
        'all': knowledge + reasoning + multilanguage + understanding,
        'knowledge': knowledge,
        'reasoning': reasoning,
        'multilanguage': multilanguage,
        'understanding': understanding
    }

    matrix_l2s_list = {
        "matrix_all_l2s": np.zeros((turn+1, 2, 2)),
        "matrix_reasoning_l2s": np.zeros((turn+1, 2, 2)),
        "matrix_understanding_l2s": np.zeros((turn+1, 2, 2)),
        "matrix_multilanguage_l2s": np.zeros((turn+1, 2, 2)),
        "matrix_knowledge_l2s": np.zeros((turn+1, 2, 2)),
    }
    
    matrix_s2s_list = {
        "matrix_all_s2s": np.zeros((turn, 2, 2)),
        "matrix_reasoning_s2s": np.zeros((turn, 2, 2)),
        "matrix_understanding_s2s": np.zeros((turn, 2, 2)),
        "matrix_multilanguage_s2s": np.zeros((turn, 2, 2)),
        "matrix_knowledge_s2s": np.zeros((turn, 2, 2)),
    }
    
    # 遍历文件夹中的 JSON 文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            for domain_name, domain in lists.items():
                json_file_path = os.path.join(folder_path, filename)
                matrix1, matrix2 = calculate_matrix(json_file_path, domain, turn)
                matrix_l2s_list[f"matrix_{domain_name}_l2s"] += matrix1
                matrix_s2s_list[f"matrix_{domain_name}_s2s"] += matrix2
                
    name = os.path.basename(folder_path)
    output1_pkl = f"main_results_count/{name}/matrix_l2s.pkl"  # 输出的 CSV 文件路径
    with open(output1_pkl, 'wb') as f:
        pickle.dump(matrix_l2s_list, f)
    output2_pkl = f"main_results_count/{name}/matrix_s2s.pkl"  # 输出的 CSV 文件路径
    with open(output2_pkl, 'wb') as f:
        pickle.dump(matrix_s2s_list, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Accept directory input")
    parser.add_argument('--wo_model', type=str, default='',required=False)
    parser.add_argument('--turn', type=int, default=3,required=False)
    args = parser.parse_args()
    turn = args.turn
    directory = "main_results"
    subfolders = [os.path.join(directory, d) for d in os.listdir(directory) 
        if os.path.isdir(os.path.join(directory, d))]
    
    for folder_path in subfolders:
        name = os.path.basename(folder_path)
        result_path = f"main_results_count/{name}"
        print(result_path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
            
        process_folder(folder_path , turn)
        generate_confushion_matrix(folder_path, turn)
        
