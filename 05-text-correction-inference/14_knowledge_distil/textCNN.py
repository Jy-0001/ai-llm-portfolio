#textCNN：
    #Config类代码：
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class Config(object):
    def __init__(self, dataset):
        self.model_name = 'textCNN'
        self.data_path = './'
        self.train_path = self.data_path + 'train.txt'
        self.dev_path = self.data_path + 'dev.txt'
        self.test_path = self.data_path + 'test.txt'
        self.class_list = [x.strip() for x in open(self.data_path + 'class.txt', encoding='utf-8').readlines()]
        self.vocab_path = self.data_path + 'vocab.pkl'
        self.save_path = './textCNN_saved_dict/saved_dict'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dropout = 0.5
        self.require_improvement = 1000
        self.num_classes = len(self.class_list) # 类别数
        self.n_vocab = 0
        self.num_epochs = 3
        self.batch_size = 128 # mini-batch大小
        self.pad_size = 32 # 每句话处理成的长度（短填长切）
        self.learning_rate = 1e-3 #学习率
        self.embed = 300 #字向量维度
        self.filter_sizes = (2,3,4) #卷积核尺寸
        self.num_filters = 512 #卷积核数量（channel数）

    #model类代码：
class model(nn.Module):
    def __init__(self,config):
        super(model,self).__init__()
        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList([nn.Conv2d(1,config.num_filters, (k,config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x
    
    def forward(self, x):
        out = self.embedding(x[0]) #文字数据都要先embedding处理
        out = out.unsqueeze(1) #适应维度
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

    #训练函数：对照知识蒸馏结构图(见课件pdf)，代码见./train_eval.py
        #导入相关工具包
        #编写获取Teacher网络输出的函数
        #编写损失值的计算函数
        #编写训练Teacher模型的训练函数
        #编写只是蒸馏的训练函数
        #编写测试函数
        #编写评估函数
    
    #运行主程序：见run.py


