# coding: UTF-8                                              # 声明源文件编码，避免中文乱码（老写法）
import numpy as np                                           # numpy：拼接数组/统计用（这里主要存预测与标签）
import torch                                                 # PyTorch：张量与训练
import torch.nn as nn                                        # nn：模块与损失函数
import torch.nn.functional as F                              # F：函数式接口（这里用F.cross_entropy）
from sklearn import metrics                                  # sklearn指标：accuracy/report/confusion等
import time                                                  # 计时
from utils import get_time_dif                               # 你utils里写的：把耗时转成 timedelta
from torch.optim import AdamW                                # 优化器 AdamW（BERT微调常用）
from tqdm import tqdm                                        # 进度条
import math                                                  # 这里没用到（可能预留）
import logging                                               # 这里没用到（可能预留）



def loss_fn(outputs, labels):                                # 自己封装一个loss函数接口
    return nn.CrossEntropyLoss()(outputs, labels)            # CrossEntropy：输入logits + int标签


def train(config, model, train_iter, dev_iter):              # 训练主函数：训练集迭代 + 验证集评估
    start_time = time.time()                                 # 记录训练开始时间
    param_optimizer = list(model.named_parameters())         # 拿到(参数名, 参数张量)列表（用于分组做weight decay）
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]# 这些参数通常不做weight decay（经验做法）
    optimizer_grouped_parameters = [                          # 构建两个参数组：做/不做weight decay
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], # 不含这些关键字的参数
            "weight_decay": 0.01                              # 对这一组加L2（AdamW里的weight_decay）
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],     # 含这些关键字的参数
            "weight_decay": 0.0                               # bias/LayerNorm不加weight_decay
        }]

    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate) # 用AdamW优化器
    total_batch = 0                                          # 计数：当前走到第几个batch（用于打印/早停）
    dev_best_loss = float("inf")                             # 当前最好的验证loss，先设无穷大，第一次必然更小
    last_improve = 0                                         # 上一次验证loss变好的batch编号
    flag = False                                             # 早停标记：True就跳出训练

    model.train()                                            # 训练模式：启用dropout、BN等训练行为
    for epoch in range(config.num_epochs):                   # 外层epoch循环
        total_batch = 0                                      # ⚠️这里每个epoch重置，会影响“跨epoch早停统计”
        print("Epoch [{}/{}]".format(epoch + 1, config.num_epochs)) # 打印当前epoch
        for i, (trains, labels) in enumerate(tqdm(train_iter)):     # train_iter每次产出(输入batch, 标签batch)
            outputs = model(trains)                          # 前向：outputs一般是logits，形状[B, num_classes]
            
            model.zero_grad()                                # 清空梯度（等价optimizer.zero_grad()也常见）
            loss = loss_fn(outputs, labels)                  # 计算训练loss（交叉熵）
            loss.backward()                                  # 反向传播：把梯度写入每个参数的 .grad
            optimizer.step()                                 # 参数更新：按梯度走一步

            if total_batch % 100 == 0 and total_batch != 0:  # 每100个batch做一次“打印+验证”
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()                     # 真实标签：搬到cpu（为了给sklearn用）
                predic = torch.max(outputs.data, 1)[1].cpu() # 预测标签：对dim=1取最大logit的索引(类别id)
                train_acc = metrics.accuracy_score(true, predic) # 训练集当前batch上的准确率
                dev_acc, dev_loss = evaluate(config, model, dev_iter) # 在验证集上跑完整评估
                if dev_loss < dev_best_loss:                 # 如果验证loss刷新历史最优
                    dev_best_loss = dev_loss                 # 更新最优loss
                    torch.save(model.state_dict(), config.save_path) # 保存参数（只存权重）
                    improve = "*"                            # 打印用：* 表示“本轮有提升”
                    last_improve = total_batch               # 记录最新一次提升发生在哪个batch
                else:
                    improve = ""                             # 没提升就空
                time_dif = get_time_dif(start_time)          # 训练已经花了多久
                msg = "Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}"
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve)) # 打印日志
                # 评估完成后将模型置于训练模式, 更新参数
                model.train()                                # evaluate里可能切成eval，这里切回train
            # 每个batch结束后累加计数
            total_batch += 1                                 # batch计数+1

            if total_batch - last_improve > config.require_improvement: # 距离上次提升超过阈值
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...") # 打印早停提示
                flag = True                                  # 标记需要停止训练
                break                                        # 跳出当前epoch内的batch循环
        if flag:                                             # 如果触发早停
            break                                            # 跳出epoch循环


def test(config, model, test_iter):                          # 测试函数：在测试集跑一次evaluate并打印报告
    # model.load_state_dict(torch.load(config.save_path))     # 如果要加载“训练保存的最好模型”，取消注释
    # 采用量化模型进行推理时需要关闭                          # 这句话意思：量化模型可能不是state_dict模式加载
    # model.eval()                                            # 若不量化，测试时一般应eval（但你这里注释掉）
    start_time = time.time()                                 # 计时：测试开始
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True) # test=True返回更多内容

    msg = "Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}"         # 输出格式
    print(msg.format(test_loss, test_acc))                    # 打印loss与acc
    print("Precision, Recall and F1-Score...")                # 提示：下面是分类报告
    print(test_report)                                        # sklearn classification_report
    print("Confusion Matrix...")                              # 提示：下面是混淆矩阵
    print(test_confusion)                                     # sklearn confusion_matrix
    time_dif = get_time_dif(start_time)                       # 测试用时
    print("Time usage:", time_dif)                            # 打印测试用时


def evaluate(config, model, data_iter, test=False):           # 评估函数：在某个数据集上算loss与acc
    # 采用量化模型进行推理时需要关闭                          # 同上：量化模型可能不方便切换eval（但一般也可以）
    model.eval()                                            # ⚠️正常评估应该eval，这里被注释，可能影响dropout一致性
    loss_total = 0                                            # 累加loss（用于求平均）
    predict_all = np.array([], dtype=int)                     # 收集所有预测类别（numpy数组）
    labels_all = np.array([], dtype=int)                      # 收集所有真实标签（numpy数组）
    with torch.no_grad():                                     # 评估时不记录梯度，省显存/更快
        for texts, labels in data_iter:                       # data_iter每次给一批：输入+标签
            outputs = model(texts)                            # 前向：logits [B, C]
            loss = F.cross_entropy(outputs, labels)           # 交叉熵loss（等价于你上面的loss_fn）

            loss_total += loss                                # 累加loss（⚠️这里加的是张量，不是loss.item()）
            labels = labels.data.cpu().numpy()                # 真实标签搬到cpu并转numpy
            predic = torch.max(outputs.data, 1)[1].cpu().numpy() # 预测类别：argmax(logits, dim=1)
            labels_all = np.append(labels_all, labels)        # 把这一批真实标签拼到总数组
            predict_all = np.append(predict_all, predic)      # 把这一批预测标签拼到总数组

    acc = metrics.accuracy_score(labels_all, predict_all)     # 用所有样本的预测/真实算准确率
    if test:                                                  # 如果是测试阶段
        report = metrics.classification_report(labels_all,predict_all,target_names=config.class_list,digits=4) # 分类报告
        confusion = metrics.confusion_matrix(labels_all, predict_all) # 混淆矩阵
        return acc, loss_total / len(data_iter), report, confusion    # 返回 acc、平均loss、report、confusion
    return acc, loss_total / len(data_iter)                   # 验证阶段只返回 acc 与平均loss
