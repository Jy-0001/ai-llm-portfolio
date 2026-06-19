import torch
from transformers import pipeline, set_seed
set_seed(55)

class Yuyuan_RedSpider():
    def __init__(self, model_path):
        self.generator = pipeline('text-generation', model=model_path + "/Yuyuan", device='cuda')
    
    def chat(self, prompt):
        response = self.generator(prompt, max_length=300, pad_token_id=50256, eos_token_id=50256, do_sample=True, num_return_sequences=1)

        return response