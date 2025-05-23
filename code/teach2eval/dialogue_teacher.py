import os
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from prompt_teacher import *
import sys
sys.path.append("..")
from model import model_path, model_use_api
from utils.function import format_conversation, generate_message, generate_responses
from utils.get_answer_api import get_answer_api
from tqdm import tqdm

def dialogue_teacher(process_id, data, gpu_ids, model_name, can_tell_answer, turn, batch_size = 128):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids 
    tensor_parallel_size = len(gpu_ids) // 2 + 1
    use_api = model_use_api[model_name]

    total_batches = (len(data) + batch_size - 1) // batch_size
    start_index_list = []
    end_index_list = []
    for batch_idx in range(total_batches):
        start_index = batch_idx * batch_size
        start_index_list.append(start_index)
        end_index = min((batch_idx + 1) * batch_size, len(data))
        end_index_list.append(end_index)
    
    whole_question_wo_options_list = [item['whole_question_wo_options'] for item in data]
    question_type_list = [item['question_type'] for item in data]
    conversation_list = [item['conversation'] for item in data]
    label_list = [item['label'] for item in data]
    options_list = [item['options'] for item in data]
    
    conversation_teacher_list = [
        format_conversation(conversation = conversation, is_student = 0)
        for conversation in conversation_list
    ]
    context = PromptTeacherContext(PromptTeacherTemplate_base())
    #生成prompt_teacher
    prompt_teacher_list = [
        context.format_prompt(whole_question_wo_options = whole_question_wo_options_list[idx], conversation_teacher = conversation_teacher_list[idx], turn = turn, can_tell_answer = can_tell_answer)
        for idx in range(len(data))
    ]
    if use_api == True:
        for idx in tqdm(range(len(data)), desc="Processing", unit="item"):
            guide = get_answer_api(prompt_teacher_list[idx],model_name)
            guide_list.append(guide)
    else:
        if "gemma" in model_name:
            raw_message_teacher_list = [
                [{"role": "user", "content": prompt_teacher}]
                for prompt_teacher in prompt_teacher_list
            ]        
        else:
            raw_message_teacher_list = [
                [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": prompt_teacher}]
                for prompt_teacher in prompt_teacher_list
            ]

        #加载tokenizer并处理message
        tokenizer = AutoTokenizer.from_pretrained(model_path[model_name],trust_remote_code=True)
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token
        stop_tokens = tokenizer.eos_token
        sampling_params = SamplingParams(max_tokens=2048, temperature=0.0, stop=stop_tokens)
        
        message_teacher_list = [
            generate_message(raw_message_teacher, tokenizer, model_name)
            for raw_message_teacher in raw_message_teacher_list
        ]

        #生成回答
        model = LLM(
            model = model_path[model_name],
            tensor_parallel_size = tensor_parallel_size,
            gpu_memory_utilization = 0.95,
            trust_remote_code=True
            )
        
        guide_list = []
        for idx in range(total_batches):
            start = start_index_list[idx]
            end = end_index_list[idx]
            responses = generate_responses(
                model = model,
                sampling_params = sampling_params,
                messages = message_teacher_list[start:end]
            )
            guide_list.extend(responses)

    for idx, guide in enumerate(guide_list):
        conversation_list[idx].append({"teacher": guide})

    for idx, conversation in enumerate(conversation_list):
        data[idx]['conversation'] = conversation
        
    return data
