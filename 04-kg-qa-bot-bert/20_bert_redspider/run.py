    #第三步：编写运行主函数
'''导入相关库'''
import time
import torch
import numpy as np
from class11_train_eval import train, test
from class11_bert import *
from importlib import import_module
import argparse
from class11utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE') #模型选择
# parser.add_argument('--model', type=str, default='class11_bert', help='choose a model: Bert, ERNIE') #模型选择
args = parser.parse_args() #解析参数

if __name__ == '__main__': #主函数
    dataset = 'RedSpider' #数据集
    if args.model =='bert': #模型选择

        model_name = 'class11_bert' #模型名称
        x = import_module(model_name) #导入模型
        config = x.Config(dataset) #构建配置文件
        np.random.seed(1) #设置随机数种子
        torch.manual_seed(1) #设置随机数种子
        torch.cuda.manual_seed_all(1) #设置随机数种子
        torch.backends.cudnn.deterministic = True #保证每次结果都一样

        print('Loading data for Bert Model') 
        train_iter, dev_iter, test_iter = build_dataset(config) #读取并预处理，返回样本list
        train_iter = build_iterator(train_iter, config) #再将样本list包装成可迭代batch流：训练流
        dev_iter = build_iterator(dev_iter, config) #验证流
        test_iter = build_iterator(test_iter, config) #测试流

        model = x.Model(config).to(config.device) #构建模型
        train(config, model, train_iter, dev_iter) #训练
        test(config, model, test_iter) #测试
    else:
        print('please assign --model')
    #调用:所在目录终端输入：python class11_bert.py --model bert