
from openai import OpenAI
import time
import unicodedata
import random
import json
import sys
sys.path.append("..")
import os

def get_answer_api(prompt, model_name):
    with open('../config.json') as f:
        config = json.load(f)
    api_key = config['api']['key']
    base_url = config['api']['base_url']
    model = model_name

    client = OpenAI(api_key=api_key, base_url=base_url)
    print("Using model:", model, "with API key:", api_key, "and base URL:", base_url)
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
    print(completion)
    message = completion.choices[0].message
    content = unicodedata.normalize('NFKC', message.content)
    return content

def main():
    # Test cases
    test_cases = [
        {
            "prompt": "What is the capital of France?",
            "model": "gemini-2.5-pro",
            "expected": "Paris"
        },
        {
            "prompt": "What is 2 + 2?",
            "model": "gemini-2.5-pro",
            "expected": "4"
        },
    ]
    
    print("Starting API tests...\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test case {i}:")
        print(f"Prompt: '{test_case['prompt']}'")
        print(f"Model: {test_case['model']}")
        print(f"Expected answer: '{test_case['expected']}'")
        
        # Get actual answer
        start_time = time.time()
        actual_answer = get_answer_api(test_case["prompt"], test_case["model"])
        elapsed_time = time.time() - start_time
        
        print(f"Actual answer: '{actual_answer}'")
        print(f"Response time: {elapsed_time:.2f} seconds")
        
        # Simple verification (case-insensitive)
        if test_case["expected"].lower() in actual_answer.lower():
            print("✅ Test passed")
        else:
            print("❌ Test failed")
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()