# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta
import os
import pickle as pkl

UNK, PAD, CLS = "[UNK]", "[PAD]", "[CLS]"  # padding符号, bert中综合信息符号
MAX_VOCAB_SIZE = 10000  # 词表长度限制

def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, "r", encoding="UTF-8") as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            content = line.split("\t")[0]

            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], 
                            key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config):
    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, "r", encoding="UTF-8") as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split("\t")
                token = config.tokenizer.tokenize(content)
                token = [CLS] + token
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)

                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += [0] * (pad_size - len(token))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size

                contents.append((token_ids, int(label), seq_len, mask))
        return contents

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test


def build_dataset_CNN(config):
    tokenizer = lambda x: [y for y in x]  # char-level
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, "rb"))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, "wb"))
    print(f"Vocab size: {len(vocab)}")

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, "r", encoding="UTF-8") as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split("\t")
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((words_line, int(label), seq_len))
        return contents  # [([...], 0), ([...], 1), ...]

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return vocab, train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device, model_name):
        self.batch_size = batch_size
        self.batches = batches
        self.model_name = model_name
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        if self.model_name == "bert" or self.model_name == "multi_task_bert":
            mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
            return (x, seq_len, mask), y
        if self.model_name == "textCNN":
            return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size : len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size : (self.index + 1) * self.batch_size]  # 从总数据 self.batches 里，按 index 取出第 index 个 batch 的样本列表（切片）
            self.index += 1                                                                             # batch 指针往后移动，下一次取下一个 batch
            batches = self._to_tensor(batches)                                                          # 把“样本列表”转换成模型要的张量 (x, y, seq_len, mask...)
            return batches                                                                              # 返回这一批张量


    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device, config.model_name)
    return iter


def get_time_dif(start_time):
    # 获取已使用时间
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

