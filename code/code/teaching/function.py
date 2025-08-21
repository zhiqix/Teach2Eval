from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import re

def str2bool(value):
    if value.lower() in ('true', '1', 't', 'y', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'f', 'n', 'no'):
        return False
    else:
        raise ValueError("Boolean value expected.")
    
def extract_guide(text):
    match = re.search(r'<guide>(.*?)</guide>', text, re.DOTALL)
    return match.group(1) if match else ""

#根据list生成对话
def format_conversation(conversation, is_student):
    new_conversation = []
    last_student_index = max(i for i, item in enumerate(conversation) if "student" in item)
    for i, item in enumerate(conversation):
            if "teacher" in item:
                text = item['teacher']
                text = extract_guide(text)
                new_conversation.append({"teacher": text})
            elif "student" in item:
                if is_student == 1:
                    if i == last_student_index:
                        new_conversation.append(item) 
                else:
                    new_conversation.append(item)
    conversation = new_conversation
    result = []
    # 给老师的部分也只保留guide部分
    for entry in conversation:
        for role, message in entry.items():
            if role == "student":
                result.append(f"Solution: {message}")
            elif role == "teacher":
                result.append(f"Guide: {message}")

    return '\n'.join(['        ' + line for line in result])

def generate_message(raw_message, tokenizer, model_name):
    message = tokenizer.apply_chat_template(raw_message, tokenize=False, add_generation_prompt = True)
    return message

def generate_responses(model, sampling_params, messages):
    outputs = model.generate(messages, sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    return responses

#用来过滤teacher回答中可能有的答案
def filter(text, str):
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