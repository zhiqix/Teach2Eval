#学生prompt 仅回答问题
prompt1_student_single_template = '''
    Your goal is to answer a multiple-choice question which has only one correct option.
    Below is the question:
    {whole_question}
    ### Instructions:
        Please carefully read the problem and options, think step by step, and select only one correct answer from the options.
        You should provide your brief thought process. At the end of your answer, return the correct choice in the format: "The answer is <your option>".
'''

prompt1_student_multiple_template = '''
    Your goal is to answer a multiple-choice question which has one or more correct answers.
    Below is the question:
    {whole_question}
    ### Instructions:
        Please carefully read the problem and options, think step by step, and select one or more correct answers from the options.
        You should provide your brief thought process. At the end of your answer, return the correct choices in the format: "The answer is <your option>, <your option>, ... <your option>".
'''

#学生prompt 对话中
prompt2_student_single_template = '''
    Your goal is to answer a multiple-choice question which has only one correct option.
    Below is the question:
    {whole_question}
    Another model will help you to improve your correctness. You will given a solution to the question, and the model will give his guidance.
    Below is the history of your conversation with the model:
    {conversation_student}
    ### Instructions:
        Please carefully read the conversation history in conjunction with the question and options, rethink step by step, and select only one correct answer from the options.
        You should provide your brief thought process. At the end of your answer, return the correct choice in the format: "The answer is <your option>".
'''

prompt2_student_multiple_template = '''
    Your goal is to answer a multiple-choice question which has one or more correct answers.
    Below is the question:
    {whole_question}
    Another model will help you to improve your correctness. You will given a solution to the question, and the model will give his guidance.
    Below is the history of your conversation with the model:
    {conversation_student}
    ### Instructions:
        Please carefully read the conversation history in conjunction with the question and options, rethink step by step, and select one or more correct answers from the options.
        You should provide your brief thought process. At the end of your answer, return the correct choices in the format: "The answer is <your option>, <your option>, ... <your option>".
'''