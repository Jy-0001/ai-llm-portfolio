import torch
from gpt2 import GPT2_RedSpider
from yuyuan import Yuyuan_RedSpider
from intern import *
from qwen import *
from deepsk import *

from config import *

class ChatGPT():
    def __init__(self, flag='deepseek', model_path="./pretrain_model"):
        # 如果调用GPT2模型的生成回复服务
        if flag == 'gpt2':
            self.generator = GPT2_RedSpider(model_path)

        # 如果调用余元模型的生成回复服务
        elif flag == 'yuyuan':
            self.generator = Yuyuan_RedSpider(model_path)

        # 如果调用大模型InternLM3-8B的⽣成回复服务
        elif flag == 'intern':
            self.generator = InternLM_RedSpider(model_path)

        # 如果调用大模型QWen2.5-1.5b的⽣成回复服务
        elif flag == 'qwen':
            self.generator = Qwen_RedSpider(model_path)

        # 如果调⽤的是DeepSeek服务
        elif flag == 'deepseek':
            self.generator = DS_RedSpider()

    def chat(self.prompt):
        res = self.generator.chat(prompt)

        return res