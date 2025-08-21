import sys
import os
sys.path.append("..")
from src.run_request import run_request
from src.check_answer import check_and_extract_answer
import re
from prompt_student import *
from prompt_teacher import *

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

def dialogue(teacher, student, strategy, can_tell_answer, item, dataset_name, max_turn = 3):
    teacher_port = 8001
    student_port = 8000
    
    index = item.get('index',None)
    question = item.get('question',None)
    options = item.get('options',[])
    passage = item.get('passage',None)
    question_type = item.get('type',-1)
    label = item.get('label',[])
    if not passage == None:
        question = passage + '\n' + question
    options_str = "Options:\n   " + "\n   ".join(options)
    whole_question = "Question: " + question + "\n" + options_str
    whole_question_wo_options = "Question: " + question
    
    result = {
        "question": question,
        "label": label,
        "options": options,
        "strategy": strategy,
        "can_tell_answer": can_tell_answer,
    }
    conversation = []
    conversation_student = ""
    conversation_teacher = ""
    if question_type == 1:
        #如果是多选题
        prompt1_student = prompt1_student_multiple_template.format(whole_question = whole_question)
    else:
        #单选题
        prompt1_student = prompt1_student_single_template.format(whole_question = whole_question)
    
    reply = run_request(student, prompt1_student, student_port)
    conversation.append({"student": reply})
    
    #print(22222)
    turn = 1
    #进行n轮对话
    while turn <= max_turn:
        #print(333,turn)
        #对话
        conversation_teacher = format_conversation(conversation = conversation, is_student = 0)
        
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
            
        prompt_teacher = context.format_prompt(whole_question_wo_options, conversation_teacher)
        if can_tell_answer == False:
            prompt_teacher += "You should not say the answer directly."
        #print(prompt_teacher)
        guide = run_request(teacher, prompt_teacher, teacher_port)
        if can_tell_answer == False:
            guide = filter(guide, "answer is")
            guide = filter(guide, "answers are")
        
        conversation.append({"teacher": guide})
        conversation_student = format_conversation(conversation = conversation, is_student = 1)
        if question_type == 1:
            #多选题
            prompt2_student = prompt2_student_multiple_template.format(whole_question = whole_question, conversation_student = conversation_student)
        else:
            prompt2_student = prompt2_student_single_template.format(whole_question = whole_question, conversation_student = conversation_student)
            
        reply = run_request(student, prompt2_student, student_port)
        conversation.append({"student": reply})
        
        #提取答案
        result[f'result{turn}_small'] = check_and_extract_answer(reply, label, options)
        turn += 1
        
    result['conversation'] = conversation
    
    return result
