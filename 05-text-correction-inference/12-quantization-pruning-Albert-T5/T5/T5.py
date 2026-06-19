# T5模型(Transfer Text-To-Text Transformer)
    # 说明：
        # 结构也采用了transformer架构，但核心思想是将所有NLP任务统一到seq2seq任务上（
        # 训练数据集：C4(Colossal Clean Crawled Corpus)
            #清洗操作：
                # 1: 只保留结尾是正常符号的⾏.
                # 2: 删除任何包含脏词汇的⻚⾯.
                # 3: 包含JavaScript词的⾏全部删除.
                # 4: 包含编程语⾔中常⽤的⼤括号的⻚⾯全部删除.
                # 5: 包含任何排版测试的⻚⾯全部删除.
                # 6: 连续三句话重复出现的情况下, 只保留⼀⾏.
    # 架构
        # 预训练策略：Encoder-Decoder(seq2seq)
            # 结构：transformer结构，包含编码器解码器
                #编码器：可以看到前面，也可以看到后面
                #解码器：只能看到前面的信息
            # MASK机制：
                #宏观角度：可以理解为自监督方法
                    #BERT风格机制：类似MLM，先将token遮掩，然后再还原出来
                #微观角度：具体对什么粒度的文本进行MASK
                    #replace span机制：可将相邻的若干token合并成一个[MASK]
            #预训练百分比策略：15%的mask比例，span为3
        #版本差异：
            #small：Encoder和Decoder都只有6层, 隐藏层的维度取512, head=8, 参数总量60 million.
            #base: Encoder和Decoder都采⽤BERT-base的参数, 参数总量220 million.
            #large: Encoder和Decoder都采⽤BERT-large的参数, 但层数保留12, 参数总量770 million.
            #3B: 在BERT-large的参数基础上, 层数采⽤24层, 参数总量3 Billion.
            #11B: 在3B参数基础上, FNN和head选取的更⼤, 参数总量11 Billion.

    #模型代码：以下皆为模型构建部分
'''config代码'''
import torch
import torch.nn as nn
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Config, T5EncoderModel

class Config(object):
    def __init__(self, dataset):
        self.model_name = 'T5' #模型名称
        self.data_path = './' #数据集路径
        self.train_path = self.data_path + 'train_dataset.txt' #训练集
        self.dev_path = self.data_path + 'dev_dataset.txt' #验证集
        self.test_path = self.data_path + 'test_dataset.txt' #测试集
        self.class_list = [x.strip() for x in open(self.data_path + 'classification.txt').readlines()]
        self.save_path = './T5_saved_dict'
        self.save_path2 = './T5_saved_dict_quantized'
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.save_path += '/' + self.model_name + '.pt' #模型训练结果保存
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#训练设备。此行代码基本不会变
        self.require_improvement = 1000 #表示若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list) #类别数
        self.num_epochs = 3 #epoch数，一般<=3
        self.batch_size = 128 #mini-batch大小
        self.pad_size = 32 #每句话处理成的长度
        self.learning_rate = 5e-5 #学习率
        self.t5_path = './t5_chinese_base'
        print(type(self.t5_path), self.t5_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.t5_path) #分词/字器
        self.t5_config = T5Config.from_pretrained(self.t5_path + '/config.json') #模型参数文件调取
#--------------------------------------------------以下为修改内容，之前为768维度-------------------------------------
        self.hidden_size = 512 #bert隐藏层维度 
#--------------------------------------------------以下为修改内容，之前为768维度-------------------------------------

        #实现model类代码：管“网络结构与 forward”（模型本体），它才是真正的神经网络，一般包括预训练底座，分类头，前向传播等
class Model(nn.Module):
    def __init__(self,config):
        super(Model, self).__init__()
        self.t5 = T5EncoderModel.from_pretrained(config.t5_path,config=config.t5_config) #模型基座
        
        # layer = ['0','1','2','3','9','10','11']
        # for name, param in self.bert.named_parameters():
        #     if name.startswith('embeddings'):
        #         print(name)
        #         param.requires_grad = False

        self.fc = nn.Linear(config.hidden_size, config.num_classes) #从隐藏层维度映射到类别数（因为要做n分类，就映射到n上）
    def forward(self, x):
        context = x[0] # 输入的句子
        mask = x[2] #对padding部分进行mask，和句子一个size，padding部分用0表示，例：[1,1,1,0,0,0]

        out = self.t5(context, attention_mask=mask) #context为数字化的文本，attention_mask为注意力掩码
        # out = self.fc(out.pooler_output) #将输出out中的pooler_output送入全连接层然后赋值给out
# -----------------------------------ai修改代码---------------------------------------------------------------------
        hidden = out.last_hidden_state                            # [B, L, d_model]

        mask_f = mask.unsqueeze(-1).float()                       # [B, L, 1]
        hidden = hidden * mask_f                                   # padding 位置变 0
        pooled = hidden.sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-9)  # [B, d_model] 逐句平均

        logits = self.fc(pooled)                                   # [B, num_classes]
        return logits
# -----------------------------------ai修改代码---------------------------------------------------------------------

        # return out
    #第二步：编写训练函数，测试函数，评估函数
'''导入相关库'''
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from utils import get_time_dif
from torch.optim import AdamW
from tqdm import tqdm
import math
import logging
        #编写训练函数
'''定义损失函数'''
def loss_fn(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)
'''编写训练函数'''
def train(config, model, train_iter, dev_iter):
    start_time = time.time() #记录训练时间
    param_optimizer = list(model.named_parameters()) #获取参数
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight'] #这些参数不做 weight decay（bias/LayerNorm），是 BERT 微调常见经验
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01}, #此组为需要权重衰减的参数，设置0.01的权重衰减
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0} #此组为不需要权重衰减的参数，设置0的权重衰减
    ] #把参数分两组，分别设置 weight_decay
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate) #选择优化器
    total_batch = 0 #记录当前训练batch
    dev_best_loss = float('inf') #inf 是 infinity（无穷大） 的缩写。首先定义最好的loss为无穷大，使得第一次的loss无论如何都满足dev_loss < dev_best_loss
    last_improve = 0 #记录上次验证集loss下降的batch数
    flag = False #记录是否很久没有效果提升

    model.train() #将模型置于训练模式
    for epoch in range(config.num_epochs):  #训练epoch数
        total_batch = 0 #记录当前epoch的batch数
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs)) #打印当前epoch
        for i, (trains, labels) in enumerate(train_iter): #训练一个epoch
            outputs = model(trains) #前向传播，outputs为模型输出的logits

            model.zero_grad()   #梯度清零
            loss = loss_fn(outputs, labels) #计算loss：outputs为模型输出的logits，labels为真实标签（正确答案）
            loss.backward() #反向传播 
            optimizer.step() #参数更新

            if total_batch % 200 == 0 and total_batch != 0:
                # 每200轮输出在训练集和验证集上的效果
                true = labels.data.cpu() #将真实标签转到cpu上
                predic = torch.max(outputs.data, 1)[1].cpu() #获取预测中最大概率的预测标签
                train_acc = metrics.accuracy_score(true, predic) #用预测和真实值做比对，算准确度
                dev_acc, dev_loss = evaluate(config, model, dev_iter) #验证集效果
                '''判断验证集的效果是否比历史最好效果更好'''
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss #记录效果最好的验证集loss
                    torch.save(model.state_dict(), config.save_path) #保存效果最好的模型
                    improve = '*' #日志提醒：此轮效果变好了
                    last_improve = total_batch #记录上次效果最好的batch数
                else:#效果没有提升时:improve=空
                    improve = ''
                time_dif = get_time_dif(start_time) #获取训练时间
                msg = 'Iter: {0:>6}, Train Loss: {1:>5.2}, Train Acc: {2:>6.2%},' \
                        'Val Loss: {3:>5.2}, Val Acc: {4:>6.2%}, Time: {5} {6}' #训练信息
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve)) #打印训练信息
                '''评估完成后将模型置于训练模式，更新参数'''
                model.train() #将模型置于训练模式
            '''每个batch结束后累加计数'''
            total_batch += 1  #每个batch结束后累加计数
            '''验证集loss超过1000batch没下降, 结束训练'''
            if total_batch - last_improve > config.require_improvement: 
                print("No optimization for a long time, auto-stopping...") 
                flag = True
                break
        if flag:
                break
        #编写测试函数
def test(config, model, test_iter):
    #model.load_state_dict(torch.load(config.save_path)) #采用量化模型进行推理时需要关闭
    model.eval() #将模型置于测试模式
    start_time = time.time() #记录测试时间
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True) #测试效果
    
    msg = 'Test Loss: {0:>5.2}, Test Acc: {1:>6.2%}' #测试信息
    print(msg.format(test_loss, test_acc)) #打印测试信息
    print("Precision, Recall and F1-Score...") #打印准确度、召回率、F1-Score
    print(test_report) #打印准确度、召回率、F1-Score
    print("Confusion Matrix...") #打印混淆矩阵

    time_dif = get_time_dif(start_time) #获取测试时间
    print("Time usage:", time_dif) #打印测试时间
        #编写验证函数：
def evaluate(config, model, data_iter, test=False):
    model.eval() #将模型置于测试模式
    loss_total = 0.0 #记录总loss
    predict_all = np.array([], dtype=int) #记录预测结果
    labels_all = np.array([], dtype=int) #记录真实结果
    with torch.no_grad(): #关闭梯度计算
        for texts, labels in data_iter: #测试一个epoch
            outputs = model(texts) #前向传播
            loss = loss_fn(outputs, labels) #计算loss

            loss_total += loss.item() #累加loss,因为损失函数为CrossEntropyLoss，返回结果为标量张量，需要item()转化成python float
            labels = labels.data.cpu().numpy() #将真实标签转为CPU上的numpy数组赋值给labels，方便用sklearn 的 metrics 计算准确率
            predict = torch.max(outputs.data, 1)[1].cpu().numpy() #获取最大概率的标签
            labels_all = np.append(labels_all, labels) 
            predict_all = np.append(predict_all, predict)

    acc = metrics.accuracy_score(labels_all, predict_all) #计算准确度
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4) #计算准确度、召回率、F1-Score
        confusion = metrics.confusion_matrix(labels_all, predict_all) #计算混淆矩阵
        return acc, loss_total / len(data_iter), report, confusion #返回准确度、loss、准确度、召回率、F1-Score、混淆矩阵
    else:
        return acc, loss_total / len(data_iter) #返回准确度和loss

    #第三步：编写运行主函数
'''导入相关库'''
import time
import torch
import numpy as np
from train_eval import train, test
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE') #模型选择
args = parser.parse_args() #解析参数

if __name__ == '__main__': #主函数
    dataset = 'toutiao' #数据集
    if args.model =='T5': #模型选择

        model_name = 'T5' #模型名称
        x = import_module(model_name) #导入模型
        config = x.Config(dataset) #构建配置文件
        np.random.seed(1) #设置随机数种子
        torch.manual_seed(1) #设置随机数种子
        torch.cuda.manual_seed_all(1) #设置随机数种子
        torch.backends.cudnn.deterministic = True #保证每次结果都一样

        print('Loading data for T5 Model') 
        train_iter, dev_iter, test_iter = build_dataset(config) #读取并预处理，返回样本list
        train_iter = build_iterator(train_iter, config) #再将样本list包装成可迭代batch流：训练流
        dev_iter = build_iterator(dev_iter, config) #验证流
        test_iter = build_iterator(test_iter, config) #测试流

        model = x.Model(config).to(config.device) #构建模型
        train(config, model, train_iter, dev_iter) #训练
        test(config, model, test_iter) #测试
    #调用:所在目录终端输入：python T5.py --model T5