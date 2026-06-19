



'''==========================训练函数实现=========================='''
# 导⼊⼯具包
import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from data import Supervised, Infer
from model import TextBackbone
from datetime import datetime
import numpy as np
import sys
import logging
import transformers
from transformers import BertModel
# from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.cuda.amp import autocast as ac
from tqdm import tqdm
# from utils import swa, FGM, PGD
from utils import swa

# 设置⽇志的相关配置
transformers.logging.set_verbosity_error()
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

# 通过命令⾏传⼊训练, 推理模式
mode = sys.argv[1]

'''重点**：对⽐学习的损失函数(有监督学习版本)'''
def sup_loss(y_pred, lamda=0.05):
    # y_pred.shape: [192, 128]
    row = torch.arange(0, y_pred.shape[0], 3, device='cuda')
    # row: tensor([ 0,  3,  6,  9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39,
    #              42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81,
    #              84, 87, 90, 93, 96, 99,102,105,108,111,114,117,120,123,
    #             126,129,132,135,138,141,144,147,150,153,156,159,162, 165,
    #             168,171,174,177,180,183,186,189]

    col = torch.arange(y_pred.shape[0], device='cuda')
    col = torch.where(col % 3 != 0)[0].cuda()
    # col: tensor([ 1,  2,  4,  5,  7,  8, 10, 11, 13, 14, 16, 17, 19, 20,
    #              22, 23, 25, 26, 28, 29, 31, 32, 34, 35, 37, 38, 40, 41,
    #              43, 44, 46, 47, 49, 50, 52, 53, 55, 56, 58, 59, 61, 62,
    #              64, 65, 67, 68, 70, 71, 73, 74, 76, 77, 79, 80, 82, 83,
    #              85, 86, 88, 89, 91, 92, 94, 95, 97, 98,100,101,103,104,
    #             106,107,109,110,112,113,115,116,118,119,121,122,124,125,
    #             127,128,130,131,133,134,136,137,139,140,142,143,145,146,
    #             148,149,151,152,154,155,157,158,160,161,163,164,166,167,
    #             169,170,172,173,175,176,178,179,181,182,184,185,187,188,
    #             190,191]

    y_true = torch.arange(0, len(col), 2, device='cuda')
    # y_true: tensor([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26,
                    # 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54,
                    # 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82,
                    # 84, 86, 88, 90, 92, 94, 96, 98,100,102,104,106,108,110,
                    #112,114,116,118,120,122,124,126]
    # y_pred.unsqueeze(1): [192, 1, 128]
    # y_pred.unsqueeze(0): [1, 192, 128]

    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    # similarities: tensor([[1.0000, 0.8282, 0.8481, ..., 0.8564, 0.8629, 0.8197],
    #                       [0.8282, 1.0000, 0.8547, ..., 0.8540, 0.8802, 0.8776],
    #                       [0.8481, 0.8547, 1.0000, ..., 0.8649, 0.8290, 0.8460],
    #                       ...,
    #                       [0.8564, 0.8540, 0.8649, ..., 1.0000, 0.8868, 0.8205],
    #                       [0.8629, 0.8802, 0.8290, ..., 0.8868, 1.0000, 0.8661],
    #                       [0.8197, 0.8776, 0.8460, ..., 0.8205, 0.8661, 1.0000]]
    # similarities.shape: [192, 192]

    similarities = torch.index_select(similarities, 0, row)
    # similarities.shape: [64, 192]

    similarities = torch.index_select(similarities, 1, col)
    # similarities.shape: [64, 128]
    # similarities: tensor([[0.8680, 0.8802, 0.8484, ..., 0.8807, 0.8630, 0.8543],
    #                       [0.8466, 0.8506, 0.8685, ..., 0.8854, 0.8488, 0.8491],
    #                       [0.8286, 0.8538, 0.6832, ..., 0.7549, 0.8460, 0.7694],
    #                       ...,
    #                       [0.8150, 0.8634, 0.8488, ..., 0.8718, 0.8361, 0.8830],
    #                       [0.8555, 0.8524, 0.8177, ..., 0.8527, 0.8384, 0.8419],
    #                       [0.7861, 0.8337, 0.8513, ..., 0.8416, 0.8129, 0.8555]]

    similarities = similarities / lamda
    loss = F.cross_entropy(similarities, y_true)
    # loss: tensor(4.9619, device='cuda:0', grad_fn=<NllLossBackward>)


    return torch.mean(loss)

def train(dataloader, model, optimizer, schedular, criterion, log_file, mode='unsup', attack_train=' '):
    # if attack_train=='fgm':
    #     fgm = FGM(model=model)

    # ⽆监督训练版本, 数据采⽤(x, x+)格式, 为⼆元组
    num = 2
    # 有监督训练版本, 数据采⽤(x, pos, neg)格式, 为三元组
    if mode == 'sup':
        num = 3

    # 很重要的⼀条, 将模型设置为训练模式
    model.train()

    all_loss = []
    # 遍历训练集数据
    for idx, data in enumerate(tqdm(dataloader)):
        # 将3个输⼊张量的shape转变成特定的格式: []
        # len(data['input_ids']) = len(data['attention_mask']) = len(data['token_type_ids']) = 64
        # data['input_ids'].shape = [64, 3, 15]
        input_ids = data['input_ids'].view(len(data['input_ids']) * num, -1).cuda()
        attention_mask = (data['attention_mask'].view(len(data['attention_mask']) * num, -1).cuda())
        token_type_ids = (data['token_type_ids'].view(len(data['token_type_ids']) * num, -1).cuda())

        # input_ids.shape = [192, 15]
        pred = model(input_ids, attention_mask, token_type_ids)
        # pred.shape = [192, 128]

        # "⽼三样"之第⼀步
        optimizer.zero_grad()
        loss = criterion(pred)
        # loss: tensor(4.6986, device='cuda:0', grad_fn=<MeanBackward0>)

        all_loss.append(loss.item())
        # "⽼三样"之第⼆步
        loss.backward()

        # "⽼三样"之第三步
        optimizer.step()
        schedular.step()

        # 每隔30个batch进⾏⼀次缓存清空, 并将信息写⼊⽇志⽂件
        if idx % 30 == 0:
            torch.cuda.empty_cache()
            with open(log_file, 'a+') as f:
                t = sum(all_loss) / len(all_loss)
                info = str(idx) + ' == {} == '.format(mode) + str(t) + '\n'
                f.write(info)
                all_loss = []

# 对⽇志⽂件做准备⼯作, 设置路径和写⼊格式
def prepare():
    os.makedirs('./output', exist_ok=True)
    now = datetime.now()
    log_file = now.strftime('%Y_%m_%d_%H_%M_%S') + '_log.txt'
    return './output/' + log_file

'''==========================主运⾏函数实现=========================='''
# ⼊⼝主函数
if __name__ == '__main__':
    # 实例化对⽐学习模型的对象model
    model = TextBackbone().cuda()

    # 训练阶段
    if mode == 'train':
        logger.info('make sup simcse train.....')
        log_file = prepare()
        
        # 准备"三元组"格式的监督学习模式数据, 进⾏对⽐学习
        dataset = Supervised()
        
        # 构建数据迭代器
        batch_size = 64
        logger.info('batch_size:{},train_num:{}'.format(batch_size, len(dataset)))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        # 设定模型参数的分模块优化策略
        param_optimizer = list(model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01,
                },
            {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
                }
            ]
        
        # 设定优化器对象
        optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)

        epochs = 10
        num_train_steps = int(len(dataloader) * epochs)
        # 设定调节器对象
        schedular = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0.05 * num_train_steps,
            num_training_steps=num_train_steps
            )
        
        # 设定损失函数, 有监督学习模式下的对⽐学习, 有独特的损失计算函数
        criterion = sup_loss

        # 外层循环epoch开启训练主流程
        for epoch in range(1, epochs + 1):
            logger.info('Epoch:{}/{}\n'.format(epoch,epochs))
            
            # 调⽤真实的训练函数
            train(dataloader, model, optimizer, schedular, criterion, log_file, mode='sup')
            
            # 每⼀个epoch轮次训练结束后, 对模型进⾏⼀次保存
            torch.save(model.state_dict(), './output/sup_model.pt')
    else:
        # 进⼊测试阶段(推理阶段)
        logger.info('make predict......')
        # 加载已经训练好的模型参数
        model.load_state_dict(torch.load('./output/sup_model.pt', map_location='cpu'), strict=True)

        # 将上⼀次存在的embedding张量⽂件删除
        if os.path.exists('doc_embedding'):
            os.remove('doc_embedding')

        # ⾮常重要的⼀步: 推理阶段将模型设置为推理模式
        model.eval()
        # 实例化推理类的对象, 将模型model作为参数传⼊
        infer = Infer(model)

        # 获取所有的股票公司的名称
        companys = infer.get_companys()

        # 写⼊embedding张量⽂件
        with open(file='doc_embedding', mode='w', encoding='utf-8') as f:
            for text in tqdm(companys):
                # 调⽤推理对象infer, 获取股票名称text的数字化张量emb
                emb = infer.get_emb(text).squeeze().detach().cpu().numpy().tolist()
                
                # 保留8位有效数字, 并转换为字符类型, ⽅便后续⽂件写⼊
                y = [str(round(i, 8)) for i in emb]
                info = text.strip() + '\t'
                info = info + ','.join(y)
                f.write(info + '\n')




