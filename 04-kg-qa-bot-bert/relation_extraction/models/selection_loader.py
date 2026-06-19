'''构建数据迭代器'''
import os
import json
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from functools import partial
from typing import Dict, List, Tuple, Set, Optional
from transformers import BertTokenizer

'''定义Selection_Dataset:这里的基类Dataset是一个抽象类, 自定义的Selection_Dataset需要继承它以及实现getitem()和len()'''
class Selection_Dataset(Dataset):
    def __init__(self, hyper, dataset):
        self.hyper = hyper # 超参数字典
        self.data_root = hyper['data_root'] # 数据目录
        self.bert_model = hyper['bert_model']
        # ===== 加载词典 =====
        self.word_vocab = json.load(open(os.path.join(self.data_root, 'word_vocab.json'), 'r'))
        self.relation_vocab = json.load(open(os.path.join(self.data_root, 'relation_vocab.json'), 'r'))
        self.bio_vocab = json.load(open(os.path.join(self.data_root, 'bio_vocab.json'), 'r'))
        # ===== 储存数据 =====
        self.selection_list = [] # 三元组选择结构
        self.text_list = [] # 原文本
        self.bio_list = [] # BIO标签
        self.spo_list = [] # 三元组真值
        # ===== bert分词器 =====
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model)
        # ===== 读取数据集，并将不用的字段分别进行加载储存 =====
        for line in open(os.path.join(self.data_root, dataset), 'r'):
            line = line.strip('\n')
            instance = json.loads(line)

            self.selection_list.append(instance['selection'])  # SPO转化的结构
            self.text_list.append(instance['text'])             # 原文本
            self.bio_list.append(instance['bio'])               # BIO标签序列
            self.spo_list.append(instance['spo_list'])          # 三元组
    '''__getitem__()''' # 对于使用bert的情况下需要自动填充
    def __getitem__(self, index):
        selection = self.selection_list[index]
        text = self.text_list[index]
        bio = self.bio_list[index]
        spo = self.spo_list[index]
        # ===== BERT预处理 =====
        if self.hyper['cell_name'] == 'bert': #判断是否用bert
            text, bio, selection = self.pad_bert(text, bio, selection)
            tokens_id = torch.tensor(self.bert_tokenizer.convert_tokens_to_ids(text))
        else:
            tokens_id = self.text2id(text) # 普通embedding
        bio_id = self.bio2id(bio) # BIO标签转id  shape: [seq_len]

        return tokens_id, bio_id, selection, len(text), spo, text, bio, self.relation_vocab
    def __len__(self):
        return len(self.text_list)

    def pad_bert(self, text, bio, selection):
        text = ['[CLS]'] + list(text) + ['[SEP]']
        bio = ['O'] + bio + ['O']
        selection = [{'subject':triplet['subject'] + 1, 'object': triplet['object'] + 
                    1, 'predicate': triplet['predicate']} for triplet in selection]
        assert len(text) <= self.hyper['max_text_len']

        text = text + ['[PAD]'] * (self.hyper['max_text_len'] - len(text))
        bio = bio + ['O'] * (self.hyper['max_text_len'] - len(bio))

        return text, bio, selection
    
    def text2id(self, text):
        oov = self.word_vocab['oov']
        text_id_list = list(map(lambda x: self.word_vocab.get(x, oov), text))
        return torch.tensor(text_id_list)
    
    def bio2id(self, bio):
        bio_id_list = list(map(lambda x: self.bio_vocab[x], bio))
        return torch.tensor(bio_id_list)

'''定义批次数据读取器，本质上是服务区collate_fn()个性化数据迭代器的函数'''
class Batch_reader(object):
    def __init__(self, data):
        data.sort(key=lambda x: len(x[0]), reverse=True)
        transposed_data = list(zip( *data)) #*data (星号解包)：如果 data 是一个二维列表（矩阵），*data 会将矩阵的每一行作为一个单独的参数传递给 zip
        self.length = transposed_data[3] # tokens_id, bio_id, selection, len(text), spo, text, bio, self.relation_vocab

        self.tokens_id = pad_sequence(transposed_data[0], batch_first=True) # 将批次中所有不定长的句子ID列表和BIO标签列表，统一填充（Pad）到相同长度，并转换成PyTorch的张量（Tensor）
        self.bio_id = pad_sequence(transposed_data[1], batch_first=True)

        batch_max_text_len = self.tokens_id.size()[1] #self.tokens_id形状：[batch_size, seq_len]
        relation_vocab = transposed_data[7][0]
        '''关系抽取模型核心步骤'''
        self.selection_id = self.selection2table(batch_max_text_len, transposed_data[2], relation_vocab)

        self.spo_gold = transposed_data[4]
        self.text = transposed_data[5]
        self.bio = transposed_data[6] 

    def pin_memory(self):
        self.tokens_id = self.tokens_id.pin_memory() # 把 CPU 张量放到“页锁定内存（pinned memory）
        self.bio_id = self.bio_id.pin_memory()
        self.selection_id = self.selection_id.pin_memory()
        return self

    '''关系抽取模型核心:selection2table函数'''
    def selection2table(self, batch_max_text_len, selection, relation_vocab):
        batch_size = len(selection) #此处传入的selection包含整个batch的三元组，所以此处len（）后代表batch_size
        
        '''初始化4维矩阵, 后续的关系抽取都采⽤4维矩阵'''
        result = torch.zeros(batch_size, batch_max_text_len, len(relation_vocab), batch_max_text_len)
        
        NA = relation_vocab['N'] # NA: 49

        for b in range(batch_size): # 外层for循环遍历批次
            result[b, :, NA, :] = 1 # ***解释：在b批次中，所有头实体与所有尾实体之间拥有NA关系（也就是没有关系   ）的都赋值成1
            for triplet in selection[b]:
                object = triplet['object']
                subject = triplet['subject']
                predicate = triplet['predicate']
                result[b, subject, predicate, object] = 1 # 有效关系设置为1, NA关系设置为0
                result[b, subject, NA, object] = 0 # 将发现关系的两者其余的关系位设置成0

        return result

def collate_fn(batch):
    return Batch_reader(batch)

Selection_loader = partial(DataLoader, collate_fn=collate_fn, pin_memory=True)











