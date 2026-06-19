# IDCNN模型：
    #代码实现：
        # 第⼀步: 配置函数的实现：见config.py
        # 第⼆步: ⼯具类函数的实现：见utils.py
        # 第三步: 模型核⼼类的实现：如下
        # 第四步: 训练代码的实现：见train.py
        # 第五步: 预测代码的实现：见train.py中的evaluate()

'''模型核⼼类的实现'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from config import *

'''构建IDCNN核心类的代码'''
#idcnn结构（这里的例子）：
    # block：四个block
        # block1
            # net1 + relu激活 + norm层
                #conv1d卷积层(dilation=1) + relu激活 + norm层
            # net2 + relu激活 + norm层
                #conv1d卷积层(dilation=1) + relu激活 + norm层
            # net3 + relu激活 + norm层
                #conv1d卷积层(dilation=3) + relu激活 + norm层
        # block2
            # net1 + relu激活 + norm层
                #conv1d卷积层(dilation=1) + relu激活 + norm层
            # net2 + relu激活 + norm层
                #conv1d卷积层(dilation=1) + relu激活 + norm层
            # net3 + relu激活 + norm层
                #conv1d卷积层(dilation=3) + relu激活 + norm层
        # block3
            # net1 + relu激活 + norm层
                #conv1d卷积层(dilation=1) + relu激活 + norm层
            # net2 + relu激活 + norm层
                #conv1d卷积层(dilation=1) + relu激活 + norm层
            # net3 + relu激活 + norm层
                #conv1d卷积层(dilation=3) + relu激活 + norm层
        # block4
            # net1 + relu激活 + norm层
                #conv1d卷积层(dilation=1) + relu激活 + norm层
            # net2 + relu激活 + norm层
                #conv1d卷积层(dilation=1) + relu激活 + norm层
            # net3 + relu激活 + norm层
                #conv1d卷积层(dilation=3) + relu激活 + norm层

class IDCNN(nn.Module):
    def __init__(self, input_size, filters, kernel_size=3, num_block=4):
        super(IDCNN, self).__init__()
        self.layers = [{'dilation': 1}, {'dilation': 1}, {'dilation': 2}]
        net = nn.Sequential()
        norms_1 = nn.ModuleList([LayerNorm(256) for _ in range(len(self.layers))]) # 封装了三个norm
        norms_2 = nn.ModuleList([LayerNorm(256) for _ in range(num_block)]) # 封装了4个norm

        '''依次构建每一层net'''
        for i in range(len(self.layers)): #遍历net数来构造net
            dilation = self.layers[i]['dilation'] #根据不同net层赋值不同dilation（膨胀系数）
            '''网络中的第一层结构是nn.Conv1d'''
            single_block = nn.Conv1d(in_channels=filters, # 有多少卷积核
                                    out_channels=filters, 
                                    kernel_size=kernel_size, # 卷积核尺寸
                                    dilation=dilation, # 赋值对应膨胀系数dilation
                                    padding=kernel_size // 2 + dilation - 1 # 做 “same padding”（卷完长度不变） 的设置,只对奇数kernal在几何上严格成立。
            )

            '''每⼀层⽹络都包含卷积层, 激活层, 正则化层, 依次添加进net中'''
            net.add_module('layer%d'%i, single_block)
            net.add_module('relu%d'%i, nn.ReLU())
            net.add_module('layernorm%d'%i, norms_1[i]) # 从norm_1中取一个norm层出来
        
        '''最后定义一个全连接层'''
        self.linear = nn.Linear(input_size, filters) # 入口维度映射到卷积核的数量
        
        '''开始构建idcnn'''
        self.idcnn = nn.Sequential()

        '''依此构建idcnn中的4个block, 每一个都包含net'''
        for i in range(num_block):
            self.idcnn.add_module('block%i'%i, net)
            self.idcnn.add_module('relu%i'%i, nn.ReLU())
            self.idcnn.add_module('layernorm%i'%i, norms_2[i])
    '''前向传播函数'''
    def forward(self, embeddings, length):
        '''首先对词嵌⼊张ᰁ进⾏全连接映射的转换'''
        embeddings = self.linear(embeddings)
        '''调整第1, 2维度'''
        embeddings = embeddings.permute(0,2,1)
        '''最后进行IDCNN的特征提取并将第2步的维度调整回原状'''
        output = self.idcnn(embeddings).permute(0,2,1)

        return output

'''采用经典transformer的LayerNorm实现策略'''
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        std = x.std(1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2




            