import time
from openai import OpenAI
def run_request(model_name, prompt, port, timeout=600, max_tokens=512, max_retries = 3):
    # 根据模型名称动态设置超时时间
    answer = ""
    base_url = f"http://localhost:{port}/v1"
    client = OpenAI(
        base_url=base_url,
        api_key="token-abc123",  # 替换为实际的API密钥
    )
    
    # 重试逻辑
    max_retries = 3  # 重试次数
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                timeout=timeout,
                max_tokens=max_tokens,
                temperature=0.0
            )
            answer = completion.choices[0].message.content
            break  # 如果请求成功，退出重试
        except Exception as e:
            if attempt < max_retries - 1:  # 如果还有剩余的重试机会
                time.sleep(3)
            else:
                print(f"error{e}")  # 如果所有重试失败，抛出异常
    return answer
