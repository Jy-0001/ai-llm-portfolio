# -*- coding: utf-8 -*-
import torch
import sys
sys.path.append('..')
import numpy as np
from transformers import BertTokenizer
from transformers import pipeline, set_seed
from transformers import AutoConfig, OPTForCausalLM, AutoTokenizer
from SimCSE.model import TextBackbone
from vllm import LLM, SamplingParams  # 高性能推理
from parser_config import parse_args
import argparse
import re
import logging
import json
import os
import pdb
import faiss
import time
# import zai

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)


class BotMedical():
    def __init__(self):
        # 初始化已训练好的SimCSE模型, 用于对NER抽取出来的待纠错entity进行文本匹配
        self.init_simcse()
        # SimCSE训练时设置的相似语义张量维度为128, 这里要和模型参数保持一致
        self.dim = 128
        # 超参数设置解析
        self.args = parse_args()
        # 初始化tokenizer
        # self.tokenizer = BertTokenizer.from_pretrained('../bert_model_data')
        self.tokenizer = BertTokenizer.from_pretrained('../SimCSE/bert-base-chinese')
        # 初始化faiss索引, 用于寻找最相似文本的加速
        self.init_index()
        # 初始化医疗问答字典
        self.init_dialog()
        # 初始化LLM聊天机制
        self.init_chat()

    def init_simcse(self):
        """加载提前训练好的SimCSE模型, 用于比较文本相似度"""
        logger.info('initialize simcse model......')
        # self.simcse_model = TextBackbone(path='./bert-base-chinese').cuda()
        self.simcse_model = TextBackbone(path='../SimCSE/bert-base-chinese').cuda()
        self.simcse_model.load_state_dict(
            torch.load('../SimCSE/output/sup_model.pt', map_location='cpu'),
            strict=True
        )
        self.simcse_model.cuda()
        self.simcse_model.eval()

    def simcse_get_emb(self, text):
        """将查询文本通过SimCSE模型预测出相似度张量"""
        text = list(text.strip())
        input = self.tokenizer.encode_plus(text, return_tensors='pt').to('cuda:0')
        emb = self.simcse_model.predict(input)
        return emb

    def faiss_search(self, text, k=3):
        """faiss检索, 返回匹配后的正确文本"""
        # 1: 得到待查询文本对应的SimCSE相似度张量
        emb = self.simcse_get_emb(text).squeeze().detach().cpu().numpy().tolist()
        emb = np.array([emb], dtype='float32')
        # 2: faiss召回TOP-K, distances越小代表越相似
        distances, results = self.index.search(emb, k)
        # distances: [[0.5357019 0.58020985 0.6094872]]
        # results:   [[5209 205 2717]]
        if float(distances[0][0]) < 0.6:
            res = self.pre_texts[int(results[0][0])]
        else:
            res = None
        return res

    def init_index(self):
        """构建faiss索引"""
        logger.info('build faiss index......')
        embeddings = []
        texts = []
        # 加载训练SimCSE模型时得到的文本相似度张量
        with open(file='../SimCSE/doc_embedding', mode='r', encoding='utf-8') as f:
            for line in f:
                text, emb = line.strip().split('\t')
                emb = [float(x) for x in emb.strip().split(',')]
                assert len(emb) == self.dim
                embeddings.append(emb)
                texts.append(text)
        embeddings = np.array(embeddings, dtype='float32')
        text2emb = {k: v for k, v in zip(texts, embeddings)}
        # faiss精确匹配索引
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(embeddings)
        # 预备查询文本的集合 / 列表
        self.texts_dict = set(texts)
        self.pre_texts = texts

    def init_dialog(self):
        """初始化医疗问答字典 {query: response}"""
        logger.info('initialize dialog dictionary......')
        path = "/root/autodl-tmp/root/41_llm_RAG/new_data/RAG_data/dialog.jsonl"
        self.dialog_dict = {}
        with open(path, mode='r', encoding='utf-8') as f:
            for line in f.readlines():
                data = json.loads(line)
                query = data['query']
                ans = data['response']
                self.dialog_dict[query] = ans

    def init_chat(self):
        """初始化LLM (vLLM)"""
        finetune_path = "/root/autodl-tmp/root/40_rlhf_on_llama/LLaMA-Factory/output/llama3_lora_dpo/"
        self.llm = LLM(
            model=finetune_path,
            tensor_parallel_size=1,
            trust_remote_code=True,
            max_model_len=512,
        )
        self.sampling_params = SamplingParams(
            temperature=1.0,
            top_p=0.95,
            max_tokens=512,
        )

    def chat(self, text):
        outputs = self.llm.generate([text], self.sampling_params)
        response = outputs[0].outputs[0].text.strip()
        return response

    def answer(self, text):
        """主入口"""
        new_text = None
        # 输入恰好是预备问题, 直接用
        if text in self.texts_dict:
            new_text = text
        # 否则做相似度匹配
        else:
            new_text = self.faiss_search(text)
        # 没匹配上, 走LLM生成
        if not new_text:
            response = self.chat(text)
        # 匹配上, 查字典直接返回预设答案
        else:
            response = self.dialog_dict[new_text]
        return response


if __name__ == '__main__':
    bot = BotMedical()
    while True:
        prompt = input('AI:')
        if prompt == 'q' or prompt == 'Q':
            break
        start_time = time.time()
        response = bot.answer(prompt)
        end_time = time.time()
        print('The dialog cost time: {}'.format(end_time - start_time))
        print('Bot:', response)