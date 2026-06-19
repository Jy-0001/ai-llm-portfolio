'''评估函数'''
def evaluate(model, dev_loader):
    # 将模型设置为评估模式
    model.eval()
    # 初始化预测列表, 标签列表
    pred = []
    gold = []
    print('evaluate')
    # 循环遍历测试集, 并评估关键指标
    for i, dev_batch in enumerate(dev_loader):
        sentence, masks, tags = dev_batch
        # 对数据进⾏Variable封装
        sentence, masks, tags = Variable(sentence), Variable(masks), Variable(tags)
        # 是否使⽤GPU进⾏加速推理
        if use_cuda:
            sentence = sentence.cuda()
            masks = masks.cuda()
            tags = tags.cuda()
        # 利⽤模型进⾏推理
        predict_tags = model(sentence, masks)


        
    # with torch.no_grad():
    #     feats = model.get_output_score(sentence, masks)   # [B,L,tagset_size+2]
    #     feats = feats[:, :, :-2]                          # ✅ 砍掉 CRF 的 start/end 两维
    #     predict_tags = feats.argmax(dim=-1)               # 现在只在真实标签空间里选





        print("unique pred:", torch.unique(predict_tags[masks.bool()]).tolist()[:20])



        # 将预测值和真实标签添加进结果列表中
        pred.extend([t for t in predict_tags.tolist()])
        gold.extend([t for t in tags.tolist()])
      
    # 将数字化标签映射回真实标签
    pred_label, gold_label = recover_label(pred, gold, l2i_dic, i2l_dic)
  
    # 计算关键指标
    acc, p, r, f = get_ner_fmeasure(gold_label, pred_label)
    print('p: {}，r: {}, f: {}'.format(p, r, f))
    # 评估结束后, 将模型设置为训练模式
    model.train()
    return acc, p, r, f









'''训练代码的实现'''
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from utils import load_vocab, load_data, recover_label, get_ner_fmeasure, save_model, load_model
from config import *
from idcnn_crf import IDCNN_CRF

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_cuda = True if torch.cuda.is_available() else False

vocab = load_vocab(vocab_file)
vocab_size = len(vocab)

'''读取训练集'''
print('max_length', max_length)
train_data = load_data(train_file, max_length=max_length, label_dic=l2i_dic, vocab=vocab)

'''提取训练数据中的3个重要字段, 并封装成LongTensor类型'''
train_ids = torch.LongTensor([temp.input_id for temp in train_data])
train_masks = torch.LongTensor([temp.input_mask for temp in train_data])
train_tags = torch.LongTensor([temp.label_id for temp in train_data])

'''标准"两步⾛", 封装数据集 + 封装迭代器'''
train_dataset = TensorDataset(train_ids, train_masks, train_tags)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

'''读取测试集'''
dev_data = load_data(dev_file, max_length=max_length, label_dic=l2i_dic, vocab=vocab)
dev_ids = torch.LongTensor([temp.input_id for temp in dev_data])
dev_masks = torch.LongTensor([temp.input_mask for temp in dev_data])
dev_tags = torch.LongTensor([temp.label_id for temp in dev_data])

dev_dataset = TensorDataset(dev_ids, dev_masks, dev_tags)
dev_loader = DataLoader(dev_dataset, shuffle=True, batch_size=batch_size)

# ====================================================================================================================================
# from transformers import BertTokenizer
# tok = BertTokenizer.from_pretrained("./bertmodel")

# sample = train_data[0]
# tokens = sample.text[:20]
# print("tokens:", tokens)
# print("your_ids:", sample.input_id[:20])
# print("bert_ids:", tok.convert_tokens_to_ids(tokens))
# ====================================================================================================================================



'''实例化模型对象model'''
model = IDCNN_CRF(vocab_size, tagset_size, 300, 64, dropout=dropout, use_cuda=use_cuda)

if use_cuda:
    model = model.cuda()

model.train()
optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.00005)

best_f = -100

'''双重for循环训练模型'''
for epoch in range(epochs):
    print('epoch: {}/{}, train'.format(epoch, epochs))
    for i, train_batch in enumerate(tqdm(train_loader)):
        sentence, masks, tags = train_batch
        sentence, masks, tags = Variable(sentence), Variable(masks), Variable(tags)

        if use_cuda:
            sentence = sentence.cuda()
            masks = masks.cuda()
            tags = tags.cuda()

        optimizer.zero_grad()
        '''训练时的损失值，需要通过调用最大似然损失计算的函数，而不是默认的forward函数'''
        loss = model.neg_log_likelihood_loss(sentence, masks, tags) #调用的是维特比损失
        loss.backward()
        optimizer.step()

    print('epoch: {}，loss: {}'.format(epoch, loss.item()))

    '''每训练完⼀个epoch, 对测试集进⾏⼀次评估'''
    acc, p, r, f = evaluate(model, dev_loader)
    '''每当有更优的F1值时, 更新最优F1, 并保存模型状态字典'''
    if f > best_f:
        torch.save(model.state_dict(), save_model_path)
        best_f = f









