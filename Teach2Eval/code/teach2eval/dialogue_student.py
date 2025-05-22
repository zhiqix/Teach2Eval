import os
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from prompt_student import *
import sys
sys.path.append("..")
from utils.check_answer import check_and_extract_answer
from model import model_path
from utils.function import format_conversation, generate_message, generate_responses

def dialogue_student(process_id, data, gpu_ids, model_name, strategy, can_tell_answer, turn, batch_size = 128):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids 
    tensor_parallel_size = len(gpu_ids) // 2 + 1
    
    total_batches = (len(data) + batch_size - 1) // batch_size
    start_index_list = []
    end_index_list = []
    for batch_idx in range(total_batches):
        start_index = batch_idx * batch_size
        start_index_list.append(start_index)
        end_index = min((batch_idx + 1) * batch_size, len(data))
        end_index_list.append(end_index)
    
    whole_question_list = [item['whole_question'] for item in data]
    question_type_list = [item['question_type'] for item in data]
    conversation_list = [item['conversation'] for item in data]
    label_list = [item['label'] for item in data]
    options_list = [item['options'] for item in data]
    
    #生成prompt_student
    if turn == 0:
        prompt_student_list = [
            prompt1_student_single_template.format(whole_question = whole_question_list[idx]) if question_type_list[idx] == 0 
            else prompt1_student_multiple_template.format(whole_question = whole_question_list[idx]) 
            for idx in range(len(data))
        ]
    else:
        conversation_student_list = [
            format_conversation(conversation = conversation, is_student = 1)
            for conversation in conversation_list
        ]
        prompt_student_list = [
            prompt2_student_single_template.format(whole_question = whole_question_list[idx], conversation_student = conversation_student_list[idx]) if question_type_list[idx] == 0 
            else prompt2_student_multiple_template.format(whole_question = whole_question_list[idx], conversation_student = conversation_student_list[idx]) 
            for idx in range(len(data))
        ]
        
    raw_message_student_list = [
        [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": prompt_student}]
        for prompt_student in prompt_student_list
    ]

    #加载tokenizer并处理message
    tokenizer = AutoTokenizer.from_pretrained(model_path[model_name],trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    stop_tokens = tokenizer.eos_token
    sampling_params = SamplingParams(max_tokens=1024, temperature=0.0, stop=stop_tokens)
    message_student_list = [
        generate_message(raw_message_student, tokenizer, model_name)
        for raw_message_student in raw_message_student_list
    ]

    #生成回答
    model = LLM(
        model = model_path[model_name],
        tensor_parallel_size = tensor_parallel_size,
        gpu_memory_utilization = 0.9,
        trust_remote_code=True
        )
    reply_list = []

    for idx in range(total_batches):
        start = start_index_list[idx]
        end = end_index_list[idx]
        responses = generate_responses(
            model = model,
            sampling_params = sampling_params,
            messages = message_student_list[start:end]
        )
        reply_list.extend(responses)

    for idx, reply in enumerate(reply_list):
        conversation_list[idx].append({"student": reply})
    
    for idx, conversation in enumerate(conversation_list):
        data[idx]['conversation'] = conversation

    for idx in range(len(data)):
        data[idx][f'result{turn}_small'] = check_and_extract_answer(reply_list[idx], label_list[idx], options_list[idx])
    
    return data
    