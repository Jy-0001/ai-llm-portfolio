'''==========================若⼲初始化类函数的实现=========================='''
import torch
import sys
sys.path.append('/root/')
sys.path.append('/root/hy-tmp/server/')
sys.path.append('/root/hy-tmp/server/infor_extract/')
import numpy as np
from transformers import BertTokenizer
from SimCSE.model import TextBackbone
from SpanPointer.src.config import Args
from SpanPointer.src.model import build_model
from utils import generate_input, span_decode, edit_distance
import logging
import json
import os
import torch
import pdb
import faiss
import time

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s', 
                    datefmt='%m/%d/%Y %H:%M:%S', 
                    level=logging.INFO)

class Correction():
    def __init__(self):
        # 1: 初始化已经训练好的NER模型, ⽤于对待纠错⽂本进⾏实体抽取
        self.init_ner()
        
        # 2: 初始化已经训练好的SimCSE模型, ⽤于对NER抽取出来的待纠错entity进⾏⽂本匹配
        self.init_simcse()
        
        # 在训练SimCSE模型的时候, 已经设置了相似语义张量的维度为128, 这⾥要和SimCSE模型参数保持⼀致
        self.dim = 128

        '''# 初始化faiss索引, ⽤于未来寻找最相似⽂本的加速'''
        self.init_index()

    '''初始化faiss index, 未来才能⽤于相似度匹配的加速'''
    def init_index(self):
        logger.info('build faiss index......')
        embeddings = []
        texts = []
        # 将训练SimCSE模型时得到的⽂本相似度张量, 以⽂件的模式加载进来
        with open(file='../SimCSE/doc_embedding', mode='r', encoding='utf-8') as f:
            for line in f:
                text, emb = line.strip().split('\t')
                # ⽂本相似度张量是128维度, 以','分隔的向量, 要转换成float类型
                emb = [float(x) for x in emb.strip().split(',')]

                # 确认⼀下读进来的张量维度, 和当初训练SimCSE的张量维度⼀致!!!
                assert len(emb) == self.dim

                embeddings.append(emb)
                texts.append(text)

        # 将⽂本和张量对应成映射字典
        embeddings = np.array(embeddings, dtype='float32')
        text2emb = {k: v for k, v in zip(texts, embeddings)}

        # 采⽤faiss的精确匹配索引模式, 建⽴索引
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(embeddings)

        # 正确的股票名称字典, 以集合的形式初始化
        self.stock_dic = set(texts)
        # stock_name属性不是⼀个股票的名称, ⽽是所有正确股票名称的总和列表
        self.stock_name = texts

    # 待纠错的⽂本text, 返回纠错后的正确⽂本以及得分
    def faiss_search(self, text, mode='distance_L', k=15):
        # 1: ⾸先得到错误⽂本text对应的SimCSE相似度张量
        # text: 俆家汇
        emb = self.simcse_get_emb(text).squeeze().detach().cpu().numpy().tolist()

        # 移动到CPU上, 转换成numpy数据类型, 再转换成list类型后, 就可以封装成numpy的array张量
        emb = np.array([emb], dtype='float32')
        # emb: emb: [[ 0.11050283 0.01384767 0.0433474 0.22737686 0.10507135 -0.07436085
        # -0.04649724 0.1542542 -0.11817002 0.01897967 -0.11548531 0.02455004
        # 0.04549803 -0.24729787 -0.0332848 -0.07336468 -0.0299028 ......]]
        # 2: 在faiss已经构建好的索引张量中, 召回TOP-K个候选结果
        # print(emb)
        _, results = self.index.search(emb, k)
        # results: [[ 867 866 4978 4977 591 3888 3889 4475 6132 9276 7677 1795 3489 6762 6287]]
        # 此处results为库中的索引

        # print("\n===== [2] FAISS search =====")
        # print("query ent:", text)          # 你传进来的 item/ent
        # print("results[0]:", results[0])
        # print("_[0]:", _[0])
        # cands = [self.stock_name[int(x)] for x in results[0]]
        # print("cands:", cands)
        # print("unique_cands:", list(dict.fromkeys(cands)))

        # 将召回的TOP-K的候选结果所对应的"正确⽂本实体ents"组装成待⽐较的列表
        ents = [self.stock_name[int(x)] for x in results[0]]
        # ents: ['徐家汇', '徐家汇', '名家汇', '名家汇', '家润多', '同花顺', '同花顺', '家家悦', '顾家家居', '合富中国', '同>庆楼', '*ST和佳', '荣之联', '御家汇', '富满微']
        
        # text: 丽⼈俪状
        # ents: ['丽⼈丽妆', '丽⼈申购', '丽尚国潮', '爱丽家居', '华丽家族', '美尔雅', '美尔雅', '美尚⽣态', '华录百纳', '华>兰⽣物', '*ST美尚', '贵⼈⻦', '美丽⽣态', '华孚时尚', '美瑞新材']
        
        # 3: 如果采⽤第⼀种模式: 经典最⼩编辑距离算法, 则默认Ranking的参数mode = 'distance_L'
        res, score = self.Ranking(text, ents, mode)
        # res: 丽⼈丽妆, score: 2

        return res, score
    
    # 将NER模型提取出来的"待纠错实体ent", 和从faiss库中搜索到的"正确的实体召回集合candidates"做评分对⽐
    def Ranking(self, ent, candidates, mode='diatance_L'):
        max_score, best_res = 10000, None

        # 遍历从faiss中召回的候选集, 计算得出最⾼分数的"正确实体"
        for candi in candidates:
            if mode == 'Levenshtein':
                score = edit_distance(ent, candi)
            # print(type(score), score)

            # 迭代更新最优分数和最优解实体
            if score < max_score:
                max_score = score
                best_res = candi

        if best_res == None:
            best_res = ent

        # 返回分数最⾼的"正确的候选实体candi", 和最⾼分数
        return best_res, max_score


    # 初始化NER模型的函数
    def init_ner(self):
        logger.info('initialize ner model')
        opt = Args().get_parser()
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')

        with open('../SpanPointer/data/span_ent2id.json', encoding='utf-8') as f:
            self.ent2id = json.load(f)
            self.id2ent = {v:k for k, v in self.ent2id.items()}

        # 在本类中, NER采⽤span指针的模式, 本质上提前训练好, 此处只做推理⽤!!!
        opt.bert_dir = './bert-base-chinese'
        self.ner_model = build_model('span', opt.bert_dir, opt, 
                                     num_tags=len(self.ent2id) + 1, 
                                     dropout_prob = opt.dropout_prob, 
                                     loss_type=opt.loss_type)
        
        # 将已经训练好的NER模型加载进来
        ner_model_path = '../SpanPointer/out/best_model.pt'
        self.ner_model.load_state_dict(torch.load(ner_model_path, map_location='cpu'), strict=True)

        # 放置到GPU上, 并设置为预测模式
        self.ner_model.cuda()
        self.ner_model.eval()

    # 初始化SimCSE模型的函数
    def init_simcse(self,):
        logger.info('initialize simcse model......')
        # 在本类中, 本质上也是将提前训练好的SimCSE模型加载进来, ⽤于⽐较⽂本相似度的预测模型来使⽤!!!
        self.simcse_model = TextBackbone().cuda()

        simcse_model_path = '../SimCSE/output/sup_model.pt'
        self.simcse_model.load_state_dict(torch.load(simcse_model_path, map_location='cpu'), strict=True)
        
        # 放置到GPU上, 并设置为预测模式
        self.simcse_model.cuda()
        self.simcse_model.eval()

    # 将待纠错的⽂本text, 通过SimCSE模型直接预测出相似度⽂本张量, 并返回
    def simcse_get_emb(self, text):
        text = list(text.strip())
        # input = self.tokenizer.encode_plus(text, return_tensors='pt').to('cuda:0')
        input = self.tokenizer(text, 
                               is_split_into_words=True,
                               truncation=True,
                               max_length=15,
                               padding="max_length",
                               return_tensors='pt', 
                              ).to('cuda:0')

        emb = self.simcse_model.predict(input)
        return emb

    '''==========================命名实体识别任务的预测实现=========================='''
    # 注意: 这是类内函数, 属于Corrector()类, 需要有代码缩进
    # 调⽤NER模型进⾏推理, 从待纠错⽂本text中提取出"候选实体", 这些"候选实体"就是"待纠错实体"
    def ner_predict(self, text):
        # 1: 调⽤tokenizer将⽂本text进⾏切割预处理
        inputs = generate_input(text, self.tokenizer)

        # 2: 调⽤NER模型执⾏实体抽取, 此处模型采⽤的span指针的模式
        decode_output = self.ner_model( **inputs)

        # 3: 获取起始位置start的概率分布, 结束位置end的概率分布
        start_logits = decode_output[0].detach().cpu().numpy()[0][1:-1]
        end_logits = decode_output[1].detach().cpu().numpy()[0][1:-1]

        # 4: 执⾏span解码函数处理, 真正的获取到从待纠错⽂本text中提取到的entities
        predict = span_decode(start_logits, end_logits, text, self.id2ent)

        return predict
    '''==========================搭建纠错类的主框架代码=========================='''
    '''
    # 注意: 这是类内函数, 属于Corrector()类, 需要有代码缩进
    # 真正进⾏⽂本纠错的函数, 此处为框架"伪代码"
    def correct(self, text, mode='distance_L'):
        # text: 待纠错的⽂本text
        # 1: 第⼀步对有错误的⽂本text执⾏NER预测, 将实体提取出来
        res = self.ner_predict(text)

        # 2: 如果没有提取出实体, 则⽂本text不需要纠错
        if not res:
            return text
        
        # 3: 遍历所有提取出来的实体
        for item in res:
            # 3.1: 如果实体本身就是"正确的股票名称", 则保留不变
            if item in '股票名称的集合':
                new_item = item
            # 3.2: 如果不在正确集合中, ⼤概率说明有问题需要纠错
            else:
                # 3.3: 直接对"有错误的实体item", 进⾏⽂本匹配, 返回"最⼤概率的正确的实体new_item"
                new_item = self.function('此函数匹配最相似的文本')

            # 3.4: ⽤"返回的最可能正确的实体new_item", 来替换"⼤概率有问题的错误实体item"
            text = text.replace(item, new_item, 1)

        # 返回纠错完毕的‘正确文本text’
        return text
    '''
    '''==========================搭建纠错类的真实代码=========================='''
    def correct(self, text, mode='distance_L'):
        # text: 待纠错的⽂本text
        # 1: 第⼀步对有错误的⽂本text执⾏NER预测, 将实体提取出来
        # 下⾯的time.time()代码, 为了测试GPU环境下的NER耗时, 单条耗时基本在15ms左右
        # 下⾯的time.time()代码, 为了测试CPU环境下的NER耗时, 单条耗时基本在50ms左右
        # start_time = time.time()
        res = self.ner_predict(text)
        res = [x for x in res if x and x.strip()] # 过滤空串
        # end_time = time.time()
        # print('The sample: {} cost time: {}'.format(text, end_time - start_time))
        
        # print("\n===== [1] NER result =====")
        # print("text:", text)
        # print("res:", res)
        # print("unique_res:", list(dict.fromkeys(res)))
        
        # 2: 如果没有提取出实体, 则⽂本text不需要纠错
        if not res:
            return text
        
        # text: 俆家汇怎么样
        # res: ['俆家汇']

        # 3: 遍历所有提取出来的实体
        for item in res:
            if not item:
                continue
            
            # 3.1: 如果实体本身就是"正确的股票名称", 则保留不变
            if item in self.stock_dic:
                new_item = item

            # 3.2: 如果不在正确集合中, ⼤概率说明有问题需要纠错
            else:
                # 3.3: 直接对"有错误的实体item", 进⾏faiss相似⽂本匹配, 返回"最⼤概率的正确的实体new_item"
                # item: 俆家汇
                new_item, score = self.faiss_search(item, mode)

            # 3.4: ⽤"返回的最可能正确的实体new_item", 来替换"⼤概率有问题的错误实体item"
            # new_item: 徐家汇
            text = text.replace(item, new_item, 1)              # <----here
            # text: 徐家汇怎么样

            # print("\n===== [3] REPLACE =====")
            # print("before:", text)
            # print("item:", item, "new_item:", new_item)
            # text = text.replace(item, new_item, 1)
            # print("after:", text)
            

        # 返回纠错完毕的"正确⽂本text"
        return text

if __name__ == '__main__':
    # 实例化纠错类Correction()j
    corr = Correction()
    with open(file='./demo.txt', mode='r', encoding='utf-8') as f:
        for line in f:
            t = line.strip()
            start_time = time.time()

            # 如果采⽤编辑距离的模式
            new_t = corr.correct(t, mode='Levenshtein')

            end_time = time.time()
            cost_time = end_time - start_time
            print('{}\t{}\t{}'.format(t, new_t, cost_time))







