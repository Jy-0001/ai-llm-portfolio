#运行主函数：
import numpy as np
import torch
import time
from train_eval import train_kd, train
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif, build_dataset_CNN

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--task', type=str, required=True, help='choose a task:trainbert, or train_kd')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'toutiao'

    if args.task == 'trainbert':
        model_name = 'bert'
        x = import_module(model_name)
        config = x. Config(dataset)
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True

        print('Loading data for Bert Model...')
        train_data, dev_data, test_data = build_dataset(config)
        train_iter = build_iterator(train_data, config)
        dev_iter = build_iterator(dev_data, config)
        test_iter = build_iterator(test_data, config)

        model = x.Model(config).to(config.device)
        train(config, model, train_iter, dev_iter, test_iter)

    if args.task == 'train_kd':
        model_name = 'bert'
        bert_module = import_module(model_name)
        bert_config = bert_module.Config(dataset)

        model_name = 'textCNN'
        cnn_module = import_module(model_name)
        cnn_config = cnn_module.Config(dataset)

        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True #保证每次结果都一样

        '''构建bert数据集'''
        bert_train_data, _, _ = build_dataset(bert_config) #因为只需要训练结果作为软⽬标，这⾥不需要dev_iter和test_iter
        bert_train_iter = build_iterator(bert_train_data, bert_config)

        '''构建cnn数据集'''
        vocab, cnn_train_data, cnn_dev_data, cnn_test_data = build_dataset_CNN(cnn_config)
        cnn_train_iter = build_iterator(cnn_train_data, cnn_config)
        cnn_dev_iter = build_iterator(cnn_dev_data, cnn_config)
        cnn_test_iter = build_iterator(cnn_test_data, cnn_config)
        cnn_config.n_vocab = len(vocab)

        print('Data loaded, now load teacher model')
        '''加载训练好的teacher模型'''
        bert_model = bert_module.Model(bert_config).to(bert_config.device)
        '''加载student模型'''
        cnn_model = cnn_module.model(cnn_config).to(cnn_config.device)

        print('Teacher and student models loaded, start training')
        train_kd(bert_config, cnn_config, bert_model, cnn_model, bert_train_iter, cnn_train_iter, cnn_dev_iter, cnn_test_iter)

