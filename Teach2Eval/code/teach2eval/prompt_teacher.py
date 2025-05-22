from abc import ABC, abstractmethod

# 策略接口：定义格式化的方法
class PromptTeacherStrategy(ABC):
    prompt_teacher_template = None
    
    
    def format_prompt(self, whole_question_wo_options, conversation_teacher, turn, can_tell_answer):
        if can_tell_answer == False:
            extra_text = '''
Please note that you should not output the final answer in any way in the guidance. 
Directly providing the correct solution and summarizing the answer is not conducive to improving another model.
In your guidance, you can use any method to point out incorrect information in the solution, such as the wrong location, reason, etc., and then provide guidance and explanation, but do not directly tell the correct solution or answer!
            '''
        else:
            extra_text = ""
        # 使用自定义的模板
        if turn == 1:
            #第一次judge，不用反思
            return self.prompt_teacher_first_turn_template.format(
                whole_question_wo_options=whole_question_wo_options,
                conversation_teacher=conversation_teacher,
                extra_text = extra_text
            )
        else:
            return self.prompt_teacher_template.format(
                whole_question_wo_options=whole_question_wo_options,
                conversation_teacher=conversation_teacher,
                extra_text = extra_text
            )

# 基础策略
class PromptTeacherTemplate_base(PromptTeacherStrategy):
    prompt_teacher_first_turn_template = '''
        Your goal is to help another model improve the accuracy of answering the question. The model will given a solution to the question, and you will give him guidance.
        Below is the question:
        {whole_question_wo_options}
        Below is the history of your conversation with the model:
        {conversation_teacher}
        Please carefully read the conversation history in conjunction with the question.
        You should first judge whether the latest solution is correct in the think process.
        Then in the guidance section, you can give a new guidance suggestion in any way you want to help the model to output the correct answer.
        {extra_text}
        Your think process and guide should be enclosed within <think> </think> and <guide> </guide> tags.
        Your output format should be:
        <think> 
        Reflection on the latest solution
        </think> 
        <guide> 
        Correctness of the latest solution: [Correct/Wrong]
        Guide (If the solution is correct, simply summarize it; if the solution is incorrect, guide the model in any way you want)
        </guide>.
    '''
    prompt_teacher_template = '''
        Your goal is to help another model improve the accuracy of answering the question. The model will given a solution to the question, and you will give him guidance.
        Below is the question:
        {whole_question_wo_options}
        Below is the history of your conversation with the model:
        {conversation_teacher}
        Please carefully read the conversation history in conjunction with the question.
        You should first judge whether the latest solution is correct in the think process.
        If you think that the latest solution is wrong, please reflect on whether there is a problem with your previous guide in the think process.
        Then in the guidance section, you can give a new guidance suggestion in any way you want to help the model to output the correct answer.
        {extra_text}
        Your think process and guide should be enclosed within <think> </think> and <guide> </guide> tags.
        Your output format should be:
        <think> 
        Reflection on the latest solution and your former guide
        </think> 
        <guide> 
        Correctness of the latest solution: [Correct/Wrong]
        Guide (If the solution is correct, simply summarize it; if the solution is incorrect, guide the model in any way you want)
        </guide>.
    '''

# 上下文：用于管理和切换策略
class PromptTeacherContext:
    def __init__(self, strategy: PromptTeacherStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: PromptTeacherStrategy):
        self._strategy = strategy

    def format_prompt(self, whole_question_wo_options, conversation_teacher, turn, can_tell_answer):
        return self._strategy.format_prompt(whole_question_wo_options, conversation_teacher, turn, can_tell_answer)
