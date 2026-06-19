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
    if args.model =='bert': #模型选择

        model_name = 'bert' #模型名称
        x = import_module(model_name) #导入模型
        config = x.Config(dataset) #构建配置文件
        np.random.seed(1) #设置随机数种子
        torch.manual_seed(1) #设置随机数种子
        torch.cuda.manual_seed_all(1) #设置随机数种子
        torch.backends.cudnn.deterministic = True #保证每次结果都一样

#===========================================================模型量化============================================================================================================
        """ 数据迭代器的预处理和生成 """
        print('Loading data for Bert Model') 
        train_iter, dev_iter, test_iter = build_dataset(config) #读取并预处理，返回样本list
        train_iter = build_iterator(train_iter, config) #再将样本list包装成可迭代batch流：训练流
        dev_iter = build_iterator(dev_iter, config) #验证流
        test_iter = build_iterator(test_iter, config) #测试流
        """ 实例化模型并加载参数 """
        model = x.Model(config)#注意与 bert 量化对比, 不能加载到GPU上, 只能在CPU上实现模型量化
        model.load_state_dict(torch.load(config.save_path))
        """ 量化BERT模型 """
        quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        print(quantized_model)
        """ 测试量化后的模型在测试集上的表现 """
        test(config, quantized_model, test_iter)
        """ 保存量化后的模型 """
        torch.save(quantized_model, config.save_path2)
    #调用:所在目录终端输入：python run.py --model bert