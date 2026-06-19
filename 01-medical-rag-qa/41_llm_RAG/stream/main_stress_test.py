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

# ====================================压力测试====================================
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import requests
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
from functools import partial

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )
# ====================================压力测试====================================

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
        # self.init_chat()

        # ====================================压力测试====================================
        # 在app.py中访问Qwen3模型服务
        self.vllm_url = "http://0.0.0.0:6006/v1/chat/completions"
        # ====================================压力测试====================================


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
        # text = list(text.strip())
        text = text.strip()
        # input = self.tokenizer._encode_plus(text, return_tensors='pt').to('cuda:0')
        input = self.tokenizer._encode_plus(
        text, 
        return_tensors='pt',
        truncation=True,         # 长 prompt 截断
        max_length=128,          # SimCSE 不需要长上下文
        padding=False
        ).to('cuda:0')
        emb = self.simcse_model.predict(input)
        return emb

    # def faiss_search(self, text, k=3):
    #     """faiss检索, 返回匹配后的正确文本"""
    #     # 1: 得到待查询文本对应的SimCSE相似度张量
    #     emb = self.simcse_get_emb(text).squeeze().detach().cpu().numpy().tolist()
    #     emb = np.array([emb], dtype='float32')
    #     # 2: faiss召回TOP-K, distances越小代表越相似
    #     distances, results = self.index.search(emb, k)
    #     # distances: [[0.5357019 0.58020985 0.6094872]]
    #     # results:   [[5209 205 2717]]
    #     if float(distances[0][0]) < 0.6:
    #         res = self.pre_texts[int(results[0][0])]
    #     else:
    #         res = None
    #     return res

    def faiss_search(self, text, k=3):
        emb = self.simcse_get_emb(text)              # 任意 shape, 但最后一维是 128
        
        # 保留最后一维(128), 把前面所有维度展平
        emb = emb.view(-1, emb.shape[-1])            # → [N, 128]
        # 多个 token embedding 做均值池化, 得到单个句向量
        emb = emb.mean(dim=0, keepdim=True)          # → [1, 128]
        
        emb = emb.detach().cpu().numpy().astype('float32')
        
        distances, results = self.index.search(emb, k)
        if float(distances[0][0]) < 0.6:
            res = self.pre_texts[int(results[0][0])]
        else:
            res = None
        return res

    def init_index(self):
        """构建faiss索引"""
        logger.info('building faiss index......')
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
        path = "/root/autodl-tmp/41_llm_RAG/new_data/RAG_data/dialog.jsonl"
        self.dialog_dict = {}
        try:
            with open(path, mode='r', encoding='utf-8') as f:
                for line in f.readlines():
                    data = json.loads(line)
                    query = data['query']
                    ans = data['response']
                    self.dialog_dict[query] = ans
            print('The number of dialog_dict:', len(self.dialog_dict.keys()))
        except FileNotFoundError:
            logger.warning("对话文件没有找到")
# ====================================压力测试====================================
# 以生成器的模式, 配合流式调用 vLLM 并解析
    def chat_stream(self, text):
        payload = {
            "model": "qwen3-4b-ppo",
            "messages": [
                    {"role": "system", "content": "你是一位非常专业的医疗助理."},
                    {"role": "user", "content": text}
                ],
            "max_tokens": 512,
            "temperature": 0.7,
            "stream": True # 开启 vLLM 流式, 只有这个开启才能测TTFT等指标
        }
        headers = {"Content-Type": "application/json"}

        try:
            # stream-True 建立长连接
            with requests.post(self.vllm_url, json=payload, headers=headers, 
                               stream=True, timeout=(5, 60)) as response:
                if response.status_code != 200:
                    yield f"Error: {response.text}"
                    return
                # 解析 SSE 流 (Server-Sent Events)
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data: "):
                            data_str = decoded_line[6:] # 去掉前缀
                            if data_str == "[DONE]":
                                break
                            try:
                                data_json = json.loads(data_str)
                                # 获取增量内容
                                delta = data_json['choices'][0]['delta']
                                if 'content' in delta:
                                    # 实时 Yield 内容给 FastAPI
                                    yield delta['content']
                            except Exception:
                                pass
        except Exception as e:
            logger.error(f"Chat Exception:{e}")
            yield "抱歉，服务暂时不可用"

    # 统一接口，异步生成器yield
    def answer_stream(self, text):
        new_text = None
        if text in self.texts_dict:
            new_text = text
        else:
            new_text = self.faiss_search(text)
        if not new_text:
            # 情况A: 走 LLM, 使用 yield from 转发流
            yield from self.chat_stream(text)
        else:
            # 情况B: 走 RAG, 直接 yield 完整结果 (模拟流)
            response = self.dialog_dict.get(new_text, "未找到对应的回答")
            yield response

# 一定要提前实例化bot类对象
bot = BotMedical()

@app.post("/")
async def server(request:Request):
    json_data = await request.json()

    if isinstance(json_data, str):
        json_data = json.loads(json_data)

    messages = json_data.get("messages", [])
    if not messages:
        return {"response": "Error: Empty messages"}
    
    prompt = messages[-1]["content"]

    # 返回 StreamingResponse
    # 注意: FastAPI 会自动将同步生成器 (answer_stream) 放入线程池运行, 不会阻塞主线程
    return StreamingResponse(
        bot.answer_stream(prompt), 
        media_type="text/plain" # 返回纯文本流, 方便压测端解析
    )

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8001)

# ====================================压力测试====================================

# 以下为原始

    # def init_chat(self):
    #     """初始化LLM (vLLM)"""
    #     finetune_path = "/root/autodl-tmp/root/40_rlhf_on_llama/LLaMA-Factory/output/llama3_lora_dpo/"
    #     self.llm = LLM(
    #         model=finetune_path,
    #         tensor_parallel_size=1,
    #         trust_remote_code=True,
    #         max_model_len=512,
    #     )
    #     self.sampling_params = SamplingParams(
    #         temperature=1.0,
    #         top_p=0.95,
    #         max_tokens=512,
    #     )

    # def chat(self, text):
    #     outputs = self.llm.generate([text], self.sampling_params)
    #     response = outputs[0].outputs[0].text.strip()
    #     return response

    # def answer(self, text):
    #     """主入口"""
    #     new_text = None
    #     # 输入恰好是预备问题, 直接用
    #     if text in self.texts_dict:
    #         new_text = text
    #     # 否则做相似度匹配
    #     else:
    #         new_text = self.faiss_search(text)
    #     # 没匹配上, 走LLM生成
    #     if not new_text:
    #         response = self.chat(text)
    #     # 匹配上, 查字典直接返回预设答案
    #     else:
    #         response = self.dialog_dict[new_text]
    #     return response


# if __name__ == '__main__':
#     bot = BotMedical()
#     while True:
#         prompt = input('AI:')
#         if prompt == 'q' or prompt == 'Q':
#             break
#         start_time = time.time()
#         response = bot.answer(prompt)
#         end_time = time.time()
#         print('The dialog cost time: {}'.format(end_time - start_time))
#         print('Bot:', response)