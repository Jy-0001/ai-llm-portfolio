import torch
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPiprline

class GPT2_RedSpider():
    def __init__(self, model_path):
        self.tokenizer = BertTokenizer.from_pretrained(model_path + '/gpt2_chinese_base')
        self.model = GPT2LMHeadModel.from_pretrained(model_path + 'gpt2_chinese_base')
        self.generator = TextGenerationPipeline(self.model, self.tokenizer)

    def chat(self, input_sentence):
        # 参数max_length不能设置的太短, ⽐如50, 那样3轮redis的对话信息就超界报错了
        output = self.generator(input_sentence, max_length=300, pad_token_id=50256, eos_token_id=50256, do_sample=True, truncation=True)

        # 为了可以限制总的文本长度，在此作阶段
        text = output[0]['generator_text'][:50]

        return text