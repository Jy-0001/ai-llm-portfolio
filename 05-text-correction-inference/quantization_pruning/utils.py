# coding: UTF-8                                           # 声明源代码文件的编码（老写法），保证中文注释/文本读取不乱码
import torch                                              # PyTorch：张量、模型、GPU等
from tqdm import tqdm                                     # 进度条工具：tqdm(iterable) 就能显示进度
import time                                               # 计时用
from datetime import timedelta                            # 把秒数转成 00:00:00 这种可读时间
import os                                                 # 路径判断/文件存在性判断
import pickle as pkl                                      # 序列化：把Python对象保存到文件/从文件读取

UNK, PAD, CLS = "[UNK]", "[PAD]", "[CLS]"                 # 三个特殊符号：未知词、填充符、BERT句首聚合符
MAX_VOCAB_SIZE = 10000                                    # 词表最大长度（主要给CNN那套手工词表用）


def build_vocab(file_path, tokenizer, max_size, min_freq):# 构建“手工词表”：统计词频→过滤→排序→编号
    vocab_dic = {}                                        # vocab_dic: {token: count} 先统计频次
    with open(file_path, "r", encoding="UTF-8") as f:      # 打开训练文本（通常每行：text\tlabel）
        for line in tqdm(f):                               # tqdm包一层，让读文件也有进度条
            line = line.strip()                            # 去掉行尾换行符和两侧空白
            if not line:                                   # 空行跳过
                continue
            content = line.split("\t")[0]                  # 取文本部分（假设第0列是文本）

            for word in tokenizer(content):                # tokenizer把文本切成token（这里是“你传进来的tokenizer”）
                vocab_dic[word] = vocab_dic.get(word, 0) + 1  # 计数：不存在就从0开始+1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], 
                            key=lambda x: x[1], reverse=True)[:max_size] # 过滤低频→按频次降序→截断到max_size
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)} # 给token编号：token→id
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1}) # 把UNK/PAD补进词表末尾
    return vocab_dic                                       # 返回：{token: id}

# 关键理解点：
# 这套 build_vocab 主要给 TextCNN 这种“自己embedding”的模型用；
# BERT/T5/ALBERT 这种“自带tokenizer+词表”的，一般不需要你自己build_vocab。


def build_dataset(config):                                 # 给 BERT 系列构建数据集（产出：token_ids + mask 等）
    def load_dataset(path, pad_size=32):                   # 读一个文件，做tokenize、padding、mask
        contents = []                                      # 最终每条样本会变成一个tuple放进contents
        with open(path, "r", encoding="UTF-8") as f:        # 打开数据文件
            for line in tqdm(f):                           # tqdm显示读取进度
                lin = line.strip()                         # 去掉换行
                if not lin:                                # 空行跳过
                    continue
                content, label = lin.split("\t")           # 假设每行：文本 \t 标签
                token = config.tokenizer.tokenize(content) # 用 HuggingFace tokenizer 做分词（输出 token 列表）
                token = [CLS] + token                      # 在开头加[CLS]：让BERT用它代表整句（常见做法）
                seq_len = len(token)                       # padding前的真实长度（用于某些模型/统计）
                mask = []                                  # attention mask：1=有效token，0=padding
                token_ids = config.tokenizer.convert_tokens_to_ids(token) # token → id（查模型词表）

                if pad_size:                               # 如果 pad_size 不是0/None，就做padding或截断
                    if len(token) < pad_size:              # 长度不够：padding到pad_size
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token)) # 前面真实token为1，后面padding为0
                        token_ids += [0] * (pad_size - len(token)) # 用0去pad（这里假设0就是[PAD]的id）
                    else:                                  # 太长：截断
                        mask = [1] * pad_size              # 截断后全是有效token，所以全1
                        token_ids = token_ids[:pad_size]   # 只保留前pad_size个
                        seq_len = pad_size                 # 截断后真实长度就当pad_size
                contents.append((token_ids, int(label), seq_len, mask)) # 一条样本：(ids, 标签, 长度, mask)
        return contents                                     # 返回样本list

    train = load_dataset(config.train_path, config.pad_size) # 构建训练集
    dev = load_dataset(config.dev_path, config.pad_size)     # 构建验证集(dev)
    test = load_dataset(config.test_path, config.pad_size)   # 构建测试集
    return train, dev, test                                  # 返回三个list

# 关键理解点：
# 1) 这里的 token_ids 其实就是“embedding要吃的索引”（embedding lookup 的 index）——来自 tokenizer+词表映射。
# 2) mask 的形状必须和 token_ids 一样长，告诉注意力：padding位置别看（注意力里会把它们屏蔽掉）。


def build_dataset_CNN(config):                               # 给 TextCNN 那套构建数据集（不用HF tokenizer）
    tokenizer = lambda x: [y for y in x]                     # char-level：把字符串拆成“单个字符”列表
    if os.path.exists(config.vocab_path):                    # 如果词表文件已存在
        vocab = pkl.load(open(config.vocab_path, "rb"))      # 直接读缓存的词表
    else:                                                    # 否则从训练集统计词频建词表
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, "wb"))       # 把词表存起来，避免下次重复统计
    print(f"Vocab size: {len(vocab)}")                       # 打印词表大小

    def load_dataset(path, pad_size=32):                     # CNN数据集加载：字符→id，并padding
        contents = []                                        # 样本list
        with open(path, "r", encoding="UTF-8") as f:          # 打开文件
            for line in tqdm(f):                             # tqdm进度条
                lin = line.strip()                           # 去换行
                if not lin:
                    continue
                content, label = lin.split("\t")             # 文本+标签
                words_line = []                              # 存放该样本的“id序列”
                token = tokenizer(content)                   # 字符列表
                seq_len = len(token)                         # padding前真实长度
                if pad_size:                                 # padding/截断
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token))) # 不够就补[PAD]字符
                    else:
                        token = token[:pad_size]             # 太长截断
                        seq_len = pad_size
                # word to id
                for word in token:                           # 每个字符查词表
                    words_line.append(vocab.get(word, vocab.get(UNK))) # 不在词表就用UNK的id
                contents.append((words_line, int(label), seq_len))    # (id序列, 标签, 长度)
        return contents                                      # 返回list：[(ids, y, len), ...]

    train = load_dataset(config.train_path, config.pad_size)  # 训练集
    dev = load_dataset(config.dev_path, config.pad_size)      # 验证集
    test = load_dataset(config.test_path, config.pad_size)    # 测试集
    return vocab, train, dev, test                            # CNN多返回一个vocab

# 关键理解点：
# CNN那套没有HuggingFace tokenizer，所以必须自己做 token→id（也就是自己做词表+embedding索引）。


class DatasetIterater(object):                                # 把“样本list”包装成“可迭代的batch流”
    def __init__(self, batches, batch_size, device, model_name):
        self.batch_size = batch_size                         # 每个batch多少条
        self.batches = batches                               # batches其实是“样本list”，不是已经分好的batch
        self.model_name = model_name                         # 用于区分bert/textCNN返回的batch结构
        self.n_batches = len(batches) // batch_size          # 完整batch的数量（整除部分）
        self.residue = False                                 # residue=True 表示最后还有“残留不满一批”的样本
        if len(batches) % self.batch_size != 0:              # 盘算残留不足一个batch的数据
            self.residue = True                              # ？？
        self.index = 0                                       # 当前走到第几个batch
        self.device = device                                 # 放到CPU还是GPU

    def _to_tensor(self, datas):                              # 把“一个batch的样本list”转换成tensor
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device) # x: [B, pad_size]，每行是token_ids
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device) # y: [B]，标签

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device) # seq_len: [B]，真实长度
        if self.model_name == "bert" or self.model_name == "multi_task_bert":
            mask = torch.LongTensor([_[3] for _ in datas]).to(self.device) # mask: [B, pad_size]
            return (x, seq_len, mask), y                      # 这就是你模型里 x[0], x[1], x[2] 的来源
        if self.model_name == "textCNN":
            return (x, seq_len), y                            # CNN不需要mask，所以只返回(x, seq_len)

    def __next__(self):                                       # Python迭代器协议：for循环会不断调用__next__()
        if self.residue and self.index == self.n_batches:     # 如果有残留，并且正好走完所有“完整batch”
            batches = self.batches[self.index * self.batch_size : len(self.batches)] # 取最后残留那一小段
            self.index += 1                                   # index往后走
            batches = self._to_tensor(batches)                # 转成tensor并放到device
            return batches                                    # 返回这一批

        elif self.index >= self.n_batches:                    # 如果完整batch也走完了（且没有残留可取）
            self.index = 0                                    # 重置index，方便下一轮epoch从头开始迭代
            raise StopIteration                               # 迭代结束信号：for循环捕获它后自然停止（不是“报错给你看”那种错）
        else:
            batches = self.batches[self.index * self.batch_size : (self.index + 1) * self.batch_size] # 取一个完整batch
            self.index += 1                                   # index加1
            batches = self._to_tensor(batches)                # 转tensor
            return batches                                    # 返回一个batch

    def __iter__(self):                                       # 让自己本身就是一个迭代器对象
        return self

    def __len__(self):                                        # len(iter) 时返回batch数量（用于tqdm显示总步数）
        if self.residue:
            return self.n_batches + 1                         # 有残留就+1
        else:
            return self.n_batches


def build_iterator(dataset, config):                          # 工厂函数：给外部用的统一接口
    iter = DatasetIterater(dataset, config.batch_size, config.device, config.model_name) # 组装迭代器
    return iter                                               # 返回可迭代对象


def get_time_dif(start_time):                                 # 计算“已用时间”
    # 获取已使用时间
    end_time = time.time()                                   # 当前时间戳（秒）
    time_dif = end_time - start_time                          # 相差多少秒
    return timedelta(seconds=int(round(time_dif)))            # 转成可读的时间差对象
