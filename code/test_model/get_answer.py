import os
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import sys
sys.path.append("..")
from utils.check_answer import check_and_extract_answer
from model import model_path, model_use_api
from utils.get_answer_api import get_answer_api
from tqdm import tqdm

def get_question(items):
    whole_question_list = []
    for item in items:
        question = item.get('question',None)
        options = item.get('options',[])
        passage = item.get('passage',None)
        if not passage == None:
            question = passage + '\n' + question
        options_str = "Options:\n   " + "\n   ".join(options)
        whole_question = "Question: " + question + "\n" + options_str
        whole_question_list.append(whole_question)
    return whole_question_list

def generate_message(raw_message, tokenizer, model_name):
    message = tokenizer.apply_chat_template(raw_message, tokenize=False, add_generation_prompt = True)
    return message

def generate_responses(model, sampling_params, messages):
    outputs = model.generate(messages, sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    return responses


prompt_single_template = '''
    You are an expert in various subjects and will be provided with a multiple-choice question.
    The question has only one correct option.
    Below is the question:
    {whole_question}
    ### Instructions:
        Please carefully read the problem and options, think step by step, and select only one correct answer from the options.
        You should provide your brief thought process. At the end of your answer, return the correct choice in the format: "The answer is <your option>".
'''

prompt_multiple_template = '''
    You are an expert in various subjects and will be provided with a multiple-choice question.
    The question has one or more correct answers.
    Below is the question:
    {whole_question}
    ### Instructions:
        Please carefully read the problem and options, think step by step, and select one or more correct answers from the options.
        You should provide your brief thought process. At the end of your answer, return the correct choices in the format: "The answer is <your option>, <your option>, ... <your option>".
'''

def get_answer(data, gpu_ids, model_name, batch_size = 128, max_turn = 3):
    #指定GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids 
    parallel_size = len(gpu_ids) // 2 + 1
    use_api = model_use_api[model_name]
    #划分batch
    total_batches = (len(data) + batch_size - 1) // batch_size
    start_index_list = []
    end_index_list = []
    for batch_idx in range(total_batches):
        start_index = batch_idx * batch_size
        start_index_list.append(start_index)
        end_index = min((batch_idx + 1) * batch_size, len(data))
        end_index_list.append(end_index)
    

    whole_question_list = get_question(data)
    index_list = [
        data[idx].get('index',-1)
        for idx in range(len(data))
    ]
    question_type_list = [
        data[idx].get('type',-1)
        for idx in range(len(data))
    ]
    options_list = [
        data[idx].get('options',None)
        for idx in range(len(data))
    ]
    label_list = [
        data[idx].get('label',[])
        for idx in range(len(data))
    ]
    dataset_name_list = [
        data[idx].get('dataset_name',None)
        for idx in range(len(data))
    ]
    
    #生成prompt_student
    prompt_student_list = [
        prompt_single_template.format(whole_question = whole_question_list[idx]) if question_type_list[idx] == 0 
        else prompt_multiple_template.format(whole_question = whole_question_list[idx]) 
        for idx in range(len(data))
    ]
    answer_list = []
    if use_api == True:
        for idx in tqdm(range(len(data)), desc="Processing", unit="item"):
            response = get_answer_api(prompt_student_list[idx],model_name)
            answer_list.append(response)
    else:
        raw_message_student_list = [
            [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": prompt_student}]
            for prompt_student in prompt_student_list
        ]
        
        #加载tokenizer并处理message
        tokenizer = AutoTokenizer.from_pretrained(model_path[model_name],trust_remote_code=True)
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token
        stop_tokens = tokenizer.eos_token
        
        max_tokens_num = 1024
            
        sampling_params = SamplingParams(max_tokens=max_tokens_num, temperature=0.0, stop=stop_tokens)
        
        message_student_list = [
            generate_message(raw_message_student, tokenizer, model_name)
            for raw_message_student in raw_message_student_list
        ]
        
        #生成回答
        model = LLM(
            model = model_path[model_name],
            tensor_parallel_size = parallel_size,
            gpu_memory_utilization=0.8,
            trust_remote_code=True
            )
        
        #for idx in tqdm(range(total_batches), desc = f"gpu:{gpu_id} turn 0: student"):
        for idx in range(total_batches):
            start = start_index_list[idx]
            end = end_index_list[idx]
            responses = generate_responses(
                model = model,
                sampling_params = sampling_params,
                messages = message_student_list[start:end]
            )
            answer_list.extend(responses)

    del model
    
    temp_result_list = [
        check_and_extract_answer(answer_list[idx], label_list[idx], options_list[idx])
        for idx in range(len(data))
    ]
    
    results = [
        {
            "dataset_name": dataset_name_list[idx],
            "index": index_list[idx],
            "result": temp_result_list[idx]
        }
        for idx in range(len(data))
    ]
    return results