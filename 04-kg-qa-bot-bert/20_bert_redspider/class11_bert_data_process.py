#基于bert预训练模型的数据处理：以投满分项目为例
    #数据预处理：
        #第⼀步: 查看项⽬数据集
            #标签文件
            #训练数据集
            #验证数据集
            #测试数据集
        #第⼆步: 查看预训练模型相关数据
            #预训练模型相关数据：
                #超参数配置文件config.json：其中包含了预训练模型的超参数设置，使用前先看
                #预训练模型文件model.bin：模型本体文件，
                #模型词典文件vocab.txt：包含词表vocab
        #第三步: 编写⼯具类函数
            #构建词表：当不使用预训练模型时，需要构建词表。（用bert的话不需要，因为词表是现成的）
import torch
from tqdm import tqdm
import time
from datetime import timedelta
import os
import pickle as pkl

UNK, PAD, CLS = "[UNK]", "[PAD]", "[CLS]"  # padding符号, bert中综合信息符号
'''构建词表'''
def build_vocab(file_path, tokenizer, max_size, min_freq): #tokenizer为序列化函数，输入文本，输出token序列
    vocab_dic = {} #词频统计字典
    '''打开词表文件'''
    with open(file_path, 'r', encoging='utf-8') as f:
        for line in tqdm(f): #利用tqdm库显示进度条
            lin = line.strip() #去掉首尾空白字符
            if not lin:#跳过空行：空行情况lin为' '，相当于False，not False就是True。
                continue
            content = lin.split('\t')[0] #取'\t'前面的内容作为文本
            for word in tokenizer(content): #把文本切成token序列
                vocab_dic[word] = vocab_dic.get(word, 0) + 1 #get()获取字典中的元素，不存在的话返回第二个输入参数
        
        vocab_list = sorted(
            [ _ for _ in vocab_dic.items() if _[1]>=min_freq], # 过滤掉频次<min_freq的词
            key=lambda x:x[1], #将vocab_dic.item()成(word, count)的列表
            reverse=True #按count从大到小排序
            )[:max_size] #取前max_size个

        vocab_dic = {word_count[0]:idx for idx, word_count in enumerate(vocab_list)} #把 vocab_list 变成 word -> idx，enumerate(vocab_list) 会得到 (idx, (word, count))
        vocab_dic.update({UNK:len(vocab_dic), PAD:len(vocab_dic) + 1}) #update将两个特殊token和词表字典拼接到一起，UNK：未登录词（unknown word），用于把“词表外的词”映射到同一个 id。PAD：填充符（padding），用于把不同长度的句子 pad 成同一长度
    return vocab_dic
            #构建数据集***：用bert的话不需要，因为bert自带分字tokenizer
                #作用：build_dataset(config) 会读取 train/dev/test 三个文件，把每一行样本处理成：
                    #token_ids：token 对应的整数 id 序列（长度固定 pad_size）
                    #label：分类标签（转成 int）
                    #seq_len：原始有效长度（没 pad 前或截断后）
                    #mask：哪些位置是“真实 token”(1)，哪些位置是“padding”(0)
                    #最后返回三个列表：train, dev, test，每个都是样本列表。
'''构建数据集（出镜率高）'''
def build_dataset(config):
    def load_dataset(path, pad_size=32): #pad_size选择整个数据集数据长度的mean + 2 * std（标准差），这样可以涵盖大部分数据。
        contents = []
        with open(path, 'r', encoding='utf-8') as f: #读取文件
            for line in tqdm(f):
                line = line.strip()
                if not line:#跳过空行
                    continue
                '''bert特有处理'''
                content, label = line.split('\t')#将一行文本解析成文本 + 标签
                token = config.tokenizer.tokenize(content)#用token变量接收转化成token的文本
                token = [CLS] + token#bert中每一个序列开头都要加'[CLS]'作分类用
                seq_len = len(token)#接收有效长度
                mask = []#初始化mask
                token_ids = config.tokenizer.convert_tokens_to_ids(token)#token转id（还没pad）
                '''bert特有处理'''
                
                '''将不同长度句子变成统一长度, 对齐batch'''
                if pad_size:
                    if len(token) < pad_size:#情况1：句子不够长，将多余token全部变为0（pad）
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += [0] * (pad_size - len(token))
                    else:#情况2：句子太长：截断句子到pad_size长度
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_ken = pad_size
                '''把样本打包进contents'''
                contents.append((token_ids, int(label), seq_len, mask))
            return contents
    train = load_dataset(config.train_path, config.pad_size)#处理训练集
    dev = load_dataset(config.dev_path, config.pad_size)#处理验证集
    test = load_dataset(config.test_path, config.pad_size)#处理测试集
    return train, dev, test

            #iterator代码：把“已经预处理好的样本列表”，包装成一个“可以被 for 循环按 batch 取、并自动转成 Tensor 的迭代器”
class DatasetIterater(object):
    def __init__(self, batches, batch_size, device, model_name):
        self.batch_size = batch_size
        self.batches = batches
        self.model_name = model_name
        self.n_batches = len(batches) // batch_size
        self.residue = False # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device
    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的⻓度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        if self.model_name == "bert" or self.model_name == "multi_task_bert":
            mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
            return (x, seq_len, mask), y

    def __next__(self):
        '''如果有残留bath, 且当前索引正好等于完整batch数'''
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size : len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        #如果已经遍历完所有batch, 重置索引并停止迭代
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        #正常情况:取一个完整batch'''
        else:
            batches = self.batches[self.index * self.batch_size : (self.index + 1) *
            self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        def __iter__(self):
            return self
        def __len__(self):
            if self.residue:
                return self.n_batches + 1
            else:
                return self.n_batches
    '''构建数据迭代器，供train/eval使用'''
    def build_iterator(dataset, config):
        iter = DatasetIterater(dataset, config.batch_size, config.device, config.model_name)
        return iter

            #get_time_dif()函数：获取已使用时间
def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

