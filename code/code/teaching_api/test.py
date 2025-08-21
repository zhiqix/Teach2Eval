
from openai import OpenAI

client = OpenAI(api_key="sk-7e5e0a919b514be0a4436ad7417a4fa6", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)

