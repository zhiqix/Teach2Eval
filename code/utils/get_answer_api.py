
from openai import OpenAI
import time
import unicodedata
import random
import json
import sys
sys.path.append("..")

def get_answer_api(prompt, model_name):

    with open('config.json') as f:
        config = json.load(f)
    api_key = config['api']['key']
    base_url = config['api']['base_url']
    model = config['api']['model']

    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # 初始化重试计数
    max_retries = 10
    retries = 0
    
    while retries < max_retries:
        try:
            # 尝试运行请求
            text = run_request(prompt, client, model)
            return text
        except Exception as e:
            # 捕获并打印异常信息
            print(f"Error occurred: {e}")
            retries += 1
            if retries < max_retries:
                print(f"Retrying... ({retries}/{max_retries})")
                time.sleep(0.5)  # 等待1秒
            else:
                print("Max retries reached.")
                return ""


def run_request(prompt, client, model):
    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        model=model,
        stream=False,
        temperature = 0,
    )

    message = completion.choices[0].message
    content = unicodedata.normalize('NFKC', message.content)
    return content

print(get_answer("What is the capital of France?", "GPT-4o"))