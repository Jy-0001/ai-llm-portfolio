from openai import OpenAI

class DS_RedSpider():
    def __init__(self):
        # 请确保在此处替换为你真实的 API Key
        self.client = OpenAI(
            api_key="sk-*********************************", 
            base_url="https://api.deepseek.com"
        )

    def chat(self, prompt):
        response = self.client.chat.completions.create(
            model="deepseek-chat", # 也可根据需求切换为 "deepseek-reasoner"
            messages=[
                {
                    "role": "system",
                    "content": "你是一个非常专业且贴心的助手"
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ],
            stream=False
        )
        return response.choices[0].message.content