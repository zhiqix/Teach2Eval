import re

def str2bool(value):
    if value.lower() in ('true', '1', 't', 'y', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'f', 'n', 'no'):
        return False
    else:
        raise ValueError("Boolean value expected.")

def remove_think_content(input_string):
    # 使用正则表达式删除<think>标签及其中的内容
    output_string = re.sub(r'<think>.*?</think>', '', input_string, flags=re.DOTALL)
    output_string = re.sub(r'<guide>|</guide>|Guide:', '', output_string)
    output_string = re.sub(r'^\s*\n', '', output_string)  # 删除开始的空行
    output_string = re.sub(r'\n\s*$', '', output_string)
    return output_string

#根据list生成对话
def format_conversation(conversation, is_student):
    new_conversation = []
    last_student_index = 0
    last_teacher_index = 0
    if is_student == 1:
        last_student_index = max(i for i, item in enumerate(conversation) if "student" in item)
        last_teacher_index = max(i for i, item in enumerate(conversation) if "teacher" in item)
    for i, item in enumerate(conversation):
            if "teacher" in item:
                text = item['teacher']
                text = remove_think_content(text)
                if is_student == 1:
                    if i == last_teacher_index:
                        new_conversation.append({"teacher": text})
                else:
                    new_conversation.append({"teacher": text})
            elif "student" in item:
                if is_student == 1:
                    if i == last_student_index:
                        new_conversation.append(item) 
                else:
                    new_conversation.append(item)
    conversation = new_conversation
    result = []

    for entry in conversation:
        for role, message in entry.items():
            if is_student == 1:
                if role == "student":
                    result.append(f"Your solution: {message}")
                elif role == "teacher":
                    result.append(f"His guide: {message}")
            else:
                if role == "student":
                    result.append(f"His solution: {message}")
                elif role == "teacher":
                    result.append(f"Your guide: {message}")
    return '\n'.join([line for line in result])

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