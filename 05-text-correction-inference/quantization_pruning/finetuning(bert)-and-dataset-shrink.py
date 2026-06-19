#模型微调：以BERT模型为例
    #构建基于BERT微调的分类模型
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertConfig
""" 构建基于BERT的微调模型类 """
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        """ 导入参数设置对象 """
        model_config = BertConfig.from_pretrained(config.bert_path, num_labels=config.num_classes)
        """ 导入基于bert-base-chinese的预训练模型 """
        self.bert = BertModel.from_pretrained(config.bert_path, config=config.bert_config)

        """ 此处用于调节是否将BERT纳入微调训练, 建议数据量+算力充足的情况下置为True """
        for param in self.bert.parameters():
            param.requires_grad = True#如果设置为False, 则保持整个BERT网络参数不变, 微调仅仅针对最后的全连接层进行训练
        """ 全连接的出口维度, 取决于具体任务 """
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        content = x[0]
        mask = x[2]

        output = self.bert(content, attention_mask=mask)
        out = self.fc(output.pooler_output)
        
        return out
    #对BERT模型的参数执行微调
        #首次展示模型中的参数命名
""" 注意: 此步需在bert.py文件中执行 """
# class Model(nn.Module):
#     def __init__(self, config):
#         super(model,self).__init__()
#         self.bert = BertModel.from_pretrained(config.bert_path, config=config.bert_config)
#         """ 将BERT中所有的参数层名字打印出来 """
#         for name, param in self.bert.named_parameters():
#             print(name)
        
#         self.fc = nn.Linear(config.hidden_size, config.num_classes)
    
    #注意：未来所有用BertModel.from_pretrained(path)模式加载进class model中的预训练模型, 都是默认所有的参数放开, 参与反向传播.
    #针对不同层微调的好处：
        #节省微调的算力
        #理解不同层的作用：
            # 1 ~ 4层: 底层偏重于字向量, 词向量
            # 5 ~ 8层: 中间层偏重于语法, 词法
            # 9 ~ 12层: 高层偏重语义, 翻译, 句法


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#用于模型微调的数据集缩小：随机删除/提取数据集使得数据集变小
""" 随机删除数据集 """
import random                               # Python标准库：随机相关函数（random.sample / random.shuffle）
from collections import defaultdict         # Python标准库：defaultdict（带“默认值工厂”的字典）

in_pathf = './'                             # 数据集所在目录（相对路径：当前目录）
out_path = './shrinked-dataset/'            # 输出目录（注意要确保这个目录已经存在，否则写文件会报错）

def stratified_sample_text(in_path, out_path, ratio=0.5, seed=42):   # 定义“分层抽样”函数：输入文件 -> 输出文件
    random.seed(seed)                       # 固定随机种子：保证每次抽样结果可复现（同seed会抽到同一批样本）

    buckets = defaultdict(list)             # buckets是“分桶字典”：key=label，value=该label下的所有行（列表）
                                            # defaultdict(list) 的关键：buckets[label] 第一次用时自动给你一个空列表 []

    with open(in_path, 'r', encoding='utf-8') as f:                  # 打开输入数据文件（逐行读取更省内存）
        for line in f:                       # 逐行读（每次拿到一行字符串）
            line = line.rstrip('\n')         # 去掉行尾换行符（只去最后的'\n'，不动其它空格）
            if not line:                     # 如果这一行变成空字符串（''），说明是空行
                continue                     # 跳过空行（continue = 直接进入下一次循环）
            text, label = line.strip().split('\t')   # 去掉两端空白后，用制表符\t切开成两段：文本 + 标签
                                                     # 注意：这里假设每行一定恰好有一个 '\t'，否则会报ValueError
            buckets[label].append(line)      # 把这一整行（原始line）放进对应label的“桶”里（按label分组完成）

    sampled = []                             # sampled 用来收集“抽样后的所有行”（跨所有label）
    for label, lines in buckets.items():     # 遍历每个桶：label是标签，lines是该标签下所有样本行
        k = max(1, int(len(lines) * ratio))  # 该标签抽多少条：桶内总数 * ratio；至少抽1条避免某类被抽没
                                            # ratio=0.5 => 每个标签保留大约一半样本（向下取整：int是截断）
        sampled.extend(random.sample(lines, k))  # random.sample：从lines里“不放回”随机抽k条，加入sampled

    random.shuffle(sampled)                  # 把最终样本顺序打乱（避免标签块状集中）
    with open(out_path, 'w', encoding='utf-8') as f:   # 打开输出文件（写入模式w会覆盖原文件）
        f.write('\n'.join(sampled) + '\n')   # 把抽样结果按行写回文件：每行之间用'\n'连接，最后再补一个换行

stratified_sample_text(in_pathf + 'train_dataset.txt', out_path + 'train_dataset.txt', ratio=0.5, seed=42)  # 抽训练集
stratified_sample_text(in_pathf + 'dev_dataset.txt',   out_path + 'dev_dataset.txt',   ratio=0.5, seed=42)  # 抽验证集
stratified_sample_text(in_pathf + 'test_dataset.txt',  out_path + 'test_dataset.txt',  ratio=0.5, seed=42)  # 抽测试集
