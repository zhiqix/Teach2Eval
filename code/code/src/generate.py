def generate_message(raw_message, tokenizer, model_name):
    message = tokenizer.apply_chat_template(raw_message, tokenize=False)
    return message

def generate_responses(model, sampling_params, messages):
    outputs = model.generate(messages, sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    return responses

def deal_prompt(prompt, model_name):
    if "Qwen" in model_name:
        prompt += "<|im_start|>assistant\n"
    return prompt