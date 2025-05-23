import re
import string

def check_and_extract_answer(text, label, options):
    #判断答案是否正确函数
    def is_answer_correct(answer, label):
        if set(answer) == set(label):
            return True
        else:
            return False

    # 提取答案函数
    def extract_answer(text):
        answer = []
        matches = re.findall(r'answer is ([A-Za-z, ]+)\.', text, re.IGNORECASE)
        answer = list(set(matches[0].replace(" ", "").split(','))) if matches else []
        return answer

    # 主流程
    #print(text)
    answer = extract_answer(text)
    text_list = text.strip().splitlines()
    if text_list == []:
        last_line = ""
    else:
        last_line = text.strip().splitlines()[-1]
    if answer == []:
        option_num = len(options)
        letters_to_check = string.ascii_uppercase[:option_num]
        answer = [letter for letter in letters_to_check if letter in last_line]
    if answer == []:
        new_options = [option[3:] for option in options]
        for i, option in enumerate(new_options):
            if option in last_line:
                answer.append(chr(65 + i))
    #print(answer)
    #print("label:   ",label)
    # 检查答案是否正确
    if is_answer_correct(answer, label):
        return (1, answer, text)
    else:
        return (0, answer, text)