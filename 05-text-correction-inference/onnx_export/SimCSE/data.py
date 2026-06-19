'''======================================数据类实现======================================'''

import json
import numpy as np
import os
from transformers import BertTokenizer
from torch.utils import data

# 构建监督学习的对⽐学习数据类代码
class Supervised(data.Dataset):
    def __init__(self, path='./data/'):
        super(Supervised, self).__init__()
        # 设置若⼲数据和参数
        self.data_dir = os.path.join(path, 'train.json')
        self.train_data = []
        self._create_train_data()
        self.company_list = self.get_list()
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')

    # 读取公司股票数据⽂件, 存⼊列表中
    def get_list(self):
        company_list = []
        with open(file='./data/股票名称.csv', mode='r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                line = [x for x in line.strip().split('\t') if x != '不详' and x != 'None' and x]

                company_list += line
        
        return company_list
    
    # 读取训练数据⽂件train.json, 采⽤source_company作为"待纠错的错误⽂本", 将target_company作为"正确的标签⽂本"
    def _create_train_data(self):
        data = json.load(open(self.data_dir, 'r', encoding='utf-8'))

        for item in data:
            source_company = item['source_company']
            target_company = item['target_company']
            # 以⼆元组的形式添加进训练集列表中
            self.train_data.append((source_company, target_company))

    # 按照索引index读取训练集数据, 本质上是通过原始⽂件中的"⼆元组", 构造出"三元组"
    # "三元组"的格式: (text, pos_text, neg_text), 例如("晋控煤⽌", "晋控煤业", "深粮控股")
    def __getitem__(self, index):
        text, pos_text = self.train_data[index]
        # 第三列的"困难负例"采⽤随机抽取⼀条股票名称数据的⽅法
        neg_text = np.random.choice(self.company_list)

        # 要确保随机抽取的股票名称和text, pos_text都不同
        while neg_text == text or neg_text == pos_text:
            neg_text = np.random.choice(self.company_list)

        # 调⽤类内函数, 对三元组进⾏tokenizer处理
        sample = self.tokenizer_process(text, pos_text, neg_text)

        return sample
    
    def tokenizer_process(self, text, pos_text, neg_text):
        sample = self.tokenizer([text, pos_text, neg_text],
                                truncation=True, 
                                add_special_token=True, 
                                max_length=15, 
                                padding='max_length', 
                                return_tensors='pt')
        
        sample = sample.to('cuda')

        return sample

    def __len__(self):
        return len(self.train_data)

'''======================================推理类实现======================================'''
class Infer():
    def __init__(self, model):
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
        self.model = model

    # 输⼊⽂本text, 获取BERT的输出张量
    def get_emb(self, text):
        # text = list(text.strip())
        text = text.strip()

        # input = self.tokenizer._encode_plus(text, return_tensors='pt').to('cuda')
        input = self.tokenizer(text, 
                               truncation=True, 
                               add_special_token=True, 
                               max_length=15, 
                               padding='max_length', 
                               return_tensors='pt').to('cuda')

        emb = self.model.predict(input)

        return emb
    
    def __getitem__(self, index):
        # text = self.all_data[index]
        # return data, text.strip()
        pass

    def __len__(self):
        # return len(self.all_data)
        pass

    def get_companys(self):
        company_list = []
        with open(file='./data/股票名称.csv', mode='r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                line = [x for x in line.strip().split('\t') if x != '不详' and x != 'None' and x]
                
                company_list += line
        
        return company_list

if __name__ == '__main__':
    Supervised()









