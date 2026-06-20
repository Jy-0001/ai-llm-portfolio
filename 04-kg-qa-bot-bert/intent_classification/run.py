"""Training entry point for BERT-based intent classification.

Usage: python run.py --model bert
"""
import logging
import argparse
from importlib import import_module

import torch
import numpy as np

from train_eval import train, test
from utils import build_dataset, build_iterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: bert')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'RedSpider'
    if args.model == 'bert':
        model_name = 'bert_model'
        x = import_module(model_name)
        config = x.Config(dataset)

        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True

        logger.info("Loading data for BERT model")
        train_data, dev_data, test_data = build_dataset(config)
        train_iter = build_iterator(train_data, config)
        dev_iter = build_iterator(dev_data, config)
        test_iter = build_iterator(test_data, config)

        model = x.Model(config).to(config.device)
        train(config, model, train_iter, dev_iter)
        test(config, model, test_iter)
    else:
        logger.error("please assign --model")
