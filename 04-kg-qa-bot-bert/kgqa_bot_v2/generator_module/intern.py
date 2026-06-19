# 需要单独安装pip install lmdeploy
import lmdeploy

class InternLM_RedSpider():
    def __init__(self, model_path):
        self.generator = lmdeploy.pipeline(model_path + "/internlm3-8b-instruct-awq")

    def chat(self, prompt):
        response = self.generator(prompt)

        return response