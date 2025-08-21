import os
from openai import OpenAI
from vllm import LLM, SamplingParams
from prompt_student import *
from prompt_teacher import *
from src.check_answer import check_and_extract_answer


def split_data(data, num_batch):
    # 将数据集分为batch
    batch_size = len(data) // num_batch
    remainder = len(data) % num_batch
    batches = []
    start = 0
    for i in range(num_batch):
        end = start + batch_size + (1 if i < remainder else 0)
        batches.append(data[start:end])
        start = end
    return batches

def get_question(items):
    whole_question_list = []
    whole_question_wo_options_list = []
    for item in items:
        question = item.get('question',None)
        options = item.get('options',[])
        passage = item.get('passage',None)
        if not passage == None:
            question = passage + '\n' + question
        options_str = "Options:\n   " + "\n   ".join(options)
        whole_question = "Question: " + question + "\n" + options_str
        whole_question_wo_options = "Question: " + question
        whole_question_list.append(whole_question)
        whole_question_wo_options_list.append(whole_question_wo_options)
    return whole_question_list, whole_question_wo_options_list

def format_conversation(conversation, is_student):
    #学生对话只保留最后一行
    if is_student == 1:
        last_student_index = max(i for i, item in enumerate(conversation) if "student" in item)
        # 构建新的列表
        new_conversation = [
            item for i, item in enumerate(conversation) 
            if "teacher" in item or (i == last_student_index and "student" in item)
        ]
        conversation = new_conversation
    result = []
    for entry in conversation:
        for role, message in entry.items():
            if is_student == 1:
                if role == "student":
                    role = "you"
                elif role == "teacher":
                    role = "the model"
            else:
                if role == "student":
                    role = "the model"
                elif role == "teacher":
                    role = "you"
            result.append(f"{role}: {message}")

    return '\n'.join(['        ' + line for line in result])

def filter(text, str):
    import re
    lines = text.splitlines()
    result_lines = []
    skip_next_two = 0
    for i in range(len(lines)):
        if skip_next_two > 0:
            skip_next_two -= 1
            continue
        if str.lower() in lines[i]:
            if i + 2 < len(lines) and re.match(r"^\*\*.*\*\*$", lines[i + 2]):
                skip_next_two = 2
            continue 

        result_lines.append(lines[i])

    return "\n".join(result_lines) 

def generate_prompt(raw_prompt, model_name):
    
    return ""

def generate_responses(model, prompt):
    stop_tokens = ["<|end_of_text|>", "</s>"]
    sampling_params = SamplingParams(max_tokens=512, temperature=0.0, stop=stop_tokens)
    outputs = model.generate(prompt, sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    return responses

def dialogue_distributed(data, gpu_ids, teacher_model_name, student_model_name, strategy, can_tell_answer, batch_size, output_file):
    result_file = os.path.join(output_file, f"result{gpu_ids[0]/2}.json")
    if os.path.exists(result_file):
        return
    
    os.environ["CUDA_VISIBLE_DEVICES"] =  str(gpu_ids[0])
    teacher_model = LLM(model = teacher_model_name, tensor_parallel_size = 1, gpu_memory_utilization = 0.95)
    # 配置第二个模型在GPU1上运行
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[1]) 
    student_model = LLM(model = student_model_name, tensor_parallel_size = 1, gpu_memory_utilization = 0.95)
    
    num_batch = (len(data) + batch_size - 1) // batch_size
    data_list = split_data(data, num_batch)
    
    results_all = []
    
    for batch in data_list:
        results = [{} for _ in range(len(batch))]
        conversation_list = [[] for _ in range(len(batch))]
        conversation_teacher = ["" for _ in range(len(batch))]
        conversation_student = ["" for _ in range(len(batch))]
        index_list = [
            batch[idx].get('index',None)
            for idx in range(len(batch))
        ]
        question_list = [
            batch[idx].get('question',None)
            for idx in range(len(batch))
        ]
        options_list = [
            batch[idx].get('options',None)
            for idx in range(len(batch))
        ]
        label_list = [
            batch[idx].get('label',[])
            for idx in range(len(batch))
        ]
        question_type_list = [
            batch[idx].get('type',-1)
            for idx in range(len(batch))
        ]
        dataset_name_list = [
            batch[idx].get('dataset_name',None)
            for idx in range(len(batch))
        ]
        result_large_list = [
            batch[idx].get('result_large',[])
            for idx in range(len(batch))
        ]
        result0_small_list = [
            batch[idx].get('result0_small',[])
            for idx in range(len(batch))
        ]
        results = [
            {
                "dataset_name": dataset_name_list[idx],
                "index": index_list[idx],
                "label": label_list[idx],
                "options": options_list[idx],
                "strategy": strategy,
                "can_tell_answer": can_tell_answer,
                "result_large": result_large_list[idx],
                "result0_small": result0_small_list[idx]
            }
            for idx in range(len(batch))
        ]
        whole_question_list, whole_question_wo_options_list = get_question(data)

        raw_prompt1_student_list = [
            prompt1_student_single_template(whole_question = whole_question) if question_type_list[idx] == 0 
            else prompt1_student_multiple_template(whole_question = whole_question) 
            for idx, whole_question in enumerate(whole_question_list)
        ]

        prompt1_student_list = [
            generate_prompt(raw_prompt1_student, student_model_name)
            for raw_prompt1_student in raw_prompt1_student_list
        ]

        reply_list = generate_responses(
            model = student_model,
            prompt = prompt1_student_list
        )
        
        for idx, reply in enumerate(reply_list):
            conversation_list[idx].append({"student": reply})
        
        turn = 1
        while turn <= max_turn:
            conversation_teacher_list = [
                format_conversation(conversation = conversation, is_student = 0)
                for conversation in conversation_list
            ]
            
            if strategy == "base":
                #基础策略
                context = PromptTeacherContext(PromptTeacherTemplate_base())
            elif strategy == "example":
                #给出例子
                context = PromptTeacherContext(PromptTeacherTemplate_example())
            elif strategy == "critique":
                #批评
                context = PromptTeacherContext(PromptTeacherTemplate_critique())
            elif strategy == "knowledge":
                #知识授予
                context = PromptTeacherContext(PromptTeacherTemplate_knowledge())
            elif strategy == "decomposition":
                #问题分解
                context = PromptTeacherContext(PromptTeacherTemplate_decomposition())
            elif strategy == "socrates":
                #苏格拉底式教育
                context = PromptTeacherContext(PromptTeacherTemplate_socrates())
            elif strategy == "all":
                context = PromptTeacherContext(PromptTeacherTemplate_all())
            else:
                #默认情况，基础策略
                context = PromptTeacherContext(PromptTeacherTemplate_base())
            
            raw_prompt_teacher_list = [
                context.format_prompt(whole_question_wo_options_list[idx], conversation_teacher_list[idx])
                for idx in range(len(batch))
            ]
            
            if can_tell_answer == False:
                raw_prompt_teacher_list = [
                    raw_prompt_teacher + "You should not say the answer directly."
                    for raw_prompt_teacher in raw_prompt_teacher_list
                ]
            
            prompt_teacher_list = [
                generate_prompt(raw_prompt_teacher, teacher_model_name)
                for raw_prompt_teacher in raw_prompt_teacher_list
            ]
            
            guide_list = generate_responses(
                model = teacher_model,
                prompt = prompt_teacher_list
            )

            if can_tell_answer == False:
                guide_list = [
                    filter(filter(guide,"answer is"),"answers are")
                    for guide in guide_list
                ]
            
            for idx, guide in enumerate(guide_list):
                conversation_list[idx].append({"teacher": guide})
            
            conversation_student_list = [
                format_conversation(conversation = conversation, is_student = 1)
                for conversation in conversation_list
            ]
            
            raw_prompt2_student_list = [
                prompt2_student_single_template(whole_question = whole_question) if batch[idx].get('type',-1) == 0 
                else prompt2_student_multiple_template(whole_question = whole_question) 
                for idx, whole_question in enumerate(whole_question_list)
            ]
            prompt2_student_list = [
                generate_prompt(raw_prompt2_student, student_model_name)
                for raw_prompt2_student in raw_prompt2_student_list
            ]
            
            reply_list = generate_responses(
                model = student_model,
                prompt = prompt2_student_list
            )
            
            temp_result_list = [
                check_and_extract_answer(reply_list[idx], label_list[idx], options_list[idx])
                for idx in range(len(batch))
            ]
            
            for idx in range(len(batch)):
                results[idx][f'result{turn}_small'] = temp_result_list[idx]
            
            turn += 1
        
        for idx in range(len(batch)):
            results[idx]['conversation'] = conversation_list[idx]
            
        results_all.extend(results)
    
    with open(result_file, 'w') as f:
        json.dump(results_all, f, indent=2)
    return