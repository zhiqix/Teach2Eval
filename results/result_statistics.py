import json
import os
import pickle
import pandas as pd
import numpy as np
import re
import copy
import argparse

knowledge = ['jecQA', 'medical_meadow_mmmlu', 'medical_meadow_medqa', 'fomc', 'financial_phrasebank', 'MMLU', 'OpenBookQA', 'ARC-E', 'MMLU-Pro', 'GPQA', 'web_questions', 'ARC-C']
multilanguage = ['AGIEval', 'mgsm', 'ceval']
understanding = ['rte', 'BigBenchLite', 'civil_comments', 'RACE', 'semeval-2014', 'QuALITY', 'imdb', 'SST2']
reasoning = ['MATH','web_of_lies', 'formal_fallacies', 'reasoning_about_colored_objects', 'navigate', 'logical_deduction', 'object_counting', 'penguins_in_a_table', 'boolean_expressions', 'aqua', 'gsm8k', 'MultiArith', 'singleq', 'svamp', 'tabmwp', 'vitaminc_fact_verification', 'emoji_movie', 'gre_reading_comprehension', 'winowhy', 'causal_judgement', 'fantasy_reasoning', 'minute_mysteries_qa', 'goal_step_wikihow', 'sports_understanding', 'disambiguation_qa', 'temporal_sequences', 'causality_mcq', 'typical_time_mcq', 'ambiguity_resolution_mcq', 'relation_mcq', 'duration_mcq', 'nli_mcq', 'frequency_mcq', 'ordering_mcq', 'date_understanding', 'arithmetic_mcq', 'storytelling_mcq']


large_model_list = []
small_model_list = ["internlm2_5-1_8b-chat", "Qwen2.5_1.5B_Instruct", "Llama-3___2-1B-Instruct", "MiniCPM-2B-dpo-bf16"]

directory = 'main_results_count'

def judge_guide(text):
    guide_content = re.search(r'<guide>(.*?)</guide>', text, re.DOTALL)
    if guide_content:
        # 提取 "Correctness of the latest solution:" 后的内容，忽略大小写
        correctness = re.search(r'correctness of the latest solution:\s*(.*?)\n', guide_content.group(1), re.IGNORECASE)
        if correctness:
            # 输出提取的内容并转换为小写
            extract_text = correctness.group(1).lower()
            if 'wrong' in extract_text:
                return 0
            elif 'correct' in extract_text:
                return 1
            else:
                return 0
        else:
            return 0
    else:
        return 0

def calculate(json_file, turn):
    # 读取 JSON 文件
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    row_names = [f'turn{i}' for i in range(turn+1)]
    col_names = ['judge_correct', 'answer_right_judge_wrong', 'correct2wrong', 'wrong2correct', 'answer_right', 'answer_wrong']
    matrix = np.zeros((len(row_names), len(col_names)))
    matrix = pd.DataFrame(matrix, columns=col_names, index=row_names)
    
    cnt_large = 0
    
    for item in data:
        if item['result_large'][0] == 1:
            cnt_large += 1
        for i in range(0,turn+1):
            if item[f'result{i}_small'][0] == 1:
                matrix.loc[f'turn{i}', 'answer_right'] += 1
            else:
                matrix.loc[f'turn{i}', 'answer_wrong'] += 1
        for i in range(0, turn):                
            text = item['conversation'][2*i+1]['teacher']
            x = judge_guide(text)
            if x == item[f'result{i}_small'][0]:
                matrix.loc[f'turn{i+1}', 'judge_correct'] += 1
            if x == 0:
                if item[f'result{i}_small'][0] == 1:
                    matrix.loc[f'turn{i+1}', 'answer_right_judge_wrong'] += 1
            # 0 1 2
            # 1 3 5
            #计算wrong2correct,correct2wrong
            if item[f'result{i}_small'][0] == 0 and item[f'result{i+1}_small'][0] == 1:
                matrix.loc[f'turn{i+1}', 'wrong2correct'] += 1
            if item[f'result{i}_small'][0] == 1 and item[f'result{i+1}_small'][0] == 0:
                matrix.loc[f'turn{i+1}', 'correct2wrong'] += 1

    delta_P = matrix.loc[f'turn{turn}', 'answer_right'] - matrix.loc['turn0', 'answer_right']
    return matrix, len(data), delta_P, cnt_large
    
def count_infomation(folder_path, turn):
    name = os.path.basename(folder_path)
    output_path = f"{directory}/{name}/statistics.pkl"  # 输出的 CSV 文件路径
    
    if os.path.exists(output_path):
        return
    
    lists = {
        'all': knowledge + reasoning + multilanguage + understanding,
        'knowledge': knowledge,
        'reasoning': reasoning,
        'multilanguage': multilanguage,
        'understanding': understanding
    }
    
    row_names = [f'turn{i}' for i in range(turn+1)]
    col_names = ['judge_correct', 'answer_right_judge_wrong', 'correct2wrong', 'wrong2correct', 'answer_right', 'answer_wrong']
    
    result_dict = {
        "matrix": np.zeros((len(row_names), len(col_names))),
        "correct_num_large": 0,
        "total_num": 0,
        "delta_P": 0,
    }
    
    result_dict['matrix'] = pd.DataFrame(result_dict['matrix'], columns=col_names, index=row_names)
    count_results = {
        "all": copy.deepcopy(result_dict),
        "reasoning": copy.deepcopy(result_dict),
        "understanding": copy.deepcopy(result_dict),
        "multilanguage": copy.deepcopy(result_dict),
        "knowledge": copy.deepcopy(result_dict),
    }
    
    for domain_name, domain in lists.items():
        # 遍历文件夹中的 JSON 文件
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                json_file_path = os.path.join(folder_path, filename)
                dataset_name = os.path.splitext(os.path.basename(json_file_path))[0]
                if dataset_name not in domain:
                    continue
                
                try:
                    matrix, total_num, delt_P, correct_num_large = calculate(json_file_path, turn)
                    count_results[domain_name]['matrix'] += matrix
                    count_results[domain_name]['total_num'] += total_num
                    count_results[domain_name]['delta_P'] += delt_P
                    count_results[domain_name]['correct_num_large'] += correct_num_large
                except Exception as e:
                    print(e)
            
    name = os.path.basename(folder_path)
    output_path = f"{directory}/{name}/statistics.pkl"  # 输出的 CSV 文件路径
    with open(output_path, 'wb') as f:
        pickle.dump(count_results, f)

def generate_matrix(statistics_path, turn, row_names, col_names):
    matrix = np.zeros((len(row_names), len(col_names)))
    matrix = pd.DataFrame(matrix, columns=col_names, index=row_names)
    if os.path.exists(statistics_path):
        print(statistics_path)
        with open(statistics_path, 'rb') as f:
            statistics_data = pickle.load(f)
        for domain_name in row_names:
            matrix.loc[domain_name, 'total_ability'] = statistics_data[domain_name]['delta_P']/statistics_data[domain_name]['total_num']
            matrix.loc[domain_name, 'answer_ability'] = statistics_data[domain_name]['correct_num_large']/statistics_data[domain_name]['total_num']
            matrix.loc[domain_name, 'judge_ability'] = statistics_data[domain_name]['matrix'].loc['turn1','judge_correct']/statistics_data[domain_name]['total_num']
            matrix.loc[domain_name, 'guide_ability'] = statistics_data[domain_name]['matrix'].loc['turn1','wrong2correct']/statistics_data[domain_name]['matrix'].loc['turn0','answer_wrong']
            matrix.loc[domain_name, 'reflect_ability'] = 1
            for i in range(2, turn+1):
                matrix.loc[domain_name, f'turn{i}_reflect_ability'] = 1 + (statistics_data[domain_name]['matrix'].loc[f'turn{i}', 'wrong2correct'] - statistics_data[domain_name]['matrix'].loc[f'turn{i}', 'correct2wrong']) / statistics_data[domain_name]['matrix'].loc[f'turn{i-1}','answer_right']
                matrix.loc[domain_name, 'reflect_ability'] *= matrix.loc[domain_name, f'turn{i}_reflect_ability']
    return matrix

# 示例调用
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Accept directory input")
    parser.add_argument('--wo_model', type=str, default='',required=False)
    parser.add_argument('--turn', type=int, default=3,required=False)
    args = parser.parse_args()
    turn = args.turn
    wo_model = args.wo_models
    directory_ori = "main_results"
    subfolders = [os.path.join(directory_ori, d) for d in os.listdir(directory_ori) 
        if os.path.isdir(os.path.join(directory_ori, d))]
    if wo_model in small_model_list:
        small_model_list.remove(wo_model)
    
    for folder_path in subfolders:
        name = os.path.basename(folder_path)
        result_path = f"{directory}/{name}"

        if not os.path.exists(result_path):
            os.makedirs(result_path)
        count_infomation(folder_path, turn)
    
    subfolders = [os.path.join(directory, d) for d in os.listdir(directory) 
        if os.path.isdir(os.path.join(directory, d))]
        
    # "Phi-3___5-mini-instruct"
    row_names = ['all', 'knowledge','multilanguage','reasoning','understanding']
    col_names = ['total_ability', 'answer_ability', 'judge_ability', 'guide_ability', 'reflect_ability']
    extra_col_names = [f'turn{i}_reflect_ability' for i in range(2,turn+1)]
    col_names.extend(extra_col_names)
    results = {}
    for large_model_name in large_model_list:
        needed_folder_path_list = []
        for folder_path in subfolders:
            name = os.path.basename(folder_path)
            if large_model_name == name.split('_teach')[0]:
                for small_model_name in small_model_list:
                    if small_model_name in name:
                        needed_folder_path_list.append(folder_path)
        tmp_matrix = [np.zeros((len(row_names), len(col_names))) for i in range(len(small_model_list))]
        for idx, folder_path in enumerate(needed_folder_path_list):
            statistics_path = os.path.join(folder_path, 'statistics.pkl')
            print(statistics_path)
            tmp_matrix[idx] = generate_matrix(statistics_path, turn, row_names, col_names)


        results[large_model_name] = np.mean(np.stack(tmp_matrix), axis=0)
        results[large_model_name] = pd.DataFrame(results[large_model_name], columns=col_names, index=row_names)
    if wo_model == '':
        output_path = f"results.pkl"  # 输出的 CSV 文件路径
    else:
        output_path = f"results_wo_{wo_model}.pkl"  # 输出的 CSV 文件路径
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
