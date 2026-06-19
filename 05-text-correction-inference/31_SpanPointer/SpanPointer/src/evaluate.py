import torch
import logging
import numpy as np
from collections import defaultdict
import pdb

logger = logging.getLogger(__name__)

# 模型评估的主代码函数, trainer.py代码⽂件中在训练主流程中调⽤此处的model_evaluate()对模型进⾏验证集上的评估
def model_evaluate(model, dev_load, config, device, ent2id):
    f = open(file='./tmp_dev_evaluate_{}'.format(config.task_type), mode='w', encoding='utf-8')
    id2ent = {v: k for k, v in ent2id.items()}
    
    # 将模型设置为评估模式
    model.eval()
    with torch.no_grad():
        for batch, batch_data in enumerate(dev_load):
            # 提取数据并最⼤程度节省GPU显存资源
            # batch_data: ['input_ids', 'token_type_ids','attention_mask', 'raw_text','start_ids','end_ids','bieo_labels']
            raw_text = batch_data['raw_text']
            del batch_data['raw_text']
            labels = batch_data['bieo_labels']
            del batch_data['bieo_labels']
            
            # 将其余有效数据信息传⾄GPU上
            for key in batch_data.keys():
                batch_data[key] = batch_data[key].to(device)
            
            # 如果解码采⽤Span Pointer模式
            if config.task_type == 'span':
                # print('111')
                decode_output = model( **batch_data)
                # print('333')
                # decode_output: (start_logits, end_logits)
                # start_logits: [64, 28, 2], end_logits: [64, 28, 2]
                start_logits = decode_output[0].cpu().numpy()
                end_logits = decode_output[1].cpu().numpy()
                # ⼀条条样本独⽴处理解码
                for tmp_start_logits, tmp_end_logits, text, label in zip(start_logits, end_logits, raw_text, labels):
                    # tmp_start_logits: [28, 2]
                    tmp_start_logits = tmp_start_logits[1: 1 + len(text)]
                    # tmp_start_logits: [12, 2]
                    # text: *st恒康重组会停牌多久
                    # len(text) = 12
                    tmp_end_logits = tmp_end_logits[1: 1 + len(text)]
                    # 预测阶段的最重要⼀步: 预测解码函数span_decode
                    predict = span_decode(tmp_start_logits, tmp_end_logits, text, id2ent)
                    tmp_label = label[:len(text)]
                    # predict: ['B-stock_name', 'I-stock_name', 'I-stock_name', 'I￾stock_name', 'E-stock_name', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
                    # tmp_label: ['B-stock_name', 'I-stock_name', 'I-stock_name', 'I￾stock_name', 'E-stock_name', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
                    # text: *st恒康重组会停牌多久

                    # 写⼊评估⽂件的数据格式分3列, (原始中⽂字符, 真实标签, 预测标签)
                    for char, true, pre in zip(text, tmp_label, predict):
                        f.write('{}\n'.format(' '.join([char, true, pre])))
                    f.write('\n')
    f.close()


# Span Pointer的核⼼解码函数, 此处采⽤"⾮重叠最短匹配策略"
def span_decode(start_logits, end_logits, raw_text, id2ent):
    predict=[]
    # start_logits: [[-2.0055513 0.7456172 ]
                #    [ 0.63194853 -1.6232361 ]
                #    [ 0.70927894 -1.6532686 ]
                #    [ 0.79790324 -1.7732315 ]
                #    [ 1.4758664 -1.7744937 ]
                #    [ 1.1061372 -1.9914815 ]
                #    [ 1.1915177 -1.9856799 ]
                #    [ 1.0427293 -1.9485211 ]
                #    [ 1.0846099 -1.9674407 ]
                #    [ 1.0224217 -1.9369218 ]
                #    [ 1.0501957 -1.8425524 ]
                #    [ 1.1026618 -1.9035641 ]]
    #
    # text: *st恒康重组会停牌多久
    
    start_pred = np.argmax(start_logits, -1)
    # start_pred: [1 0 0 0 0 0 0 0 0 0 0 0]
    end_pred = np.argmax(end_logits, -1)
    # end_pred: [0 0 0 0 1 0 0 0 0 0 0 0]
    
    # 循环解码综合考虑不同的标签 种类s_type, 和不同的下标i
    for i, s_type in enumerate(start_pred):
        if s_type == 0:
            continue
        for j, e_type in enumerate(end_pred[i:]):
            if s_type == e_type:
                tmp_ent = raw_text[i: i + j + 1]
                predict.append((''.join(tmp_ent), i, i + j, s_type))
                
                # 因为最短匹配的策略，所以直接break结束内循环
                break
    # predict: [('*st恒康', 0, 4, 1)]
    
    # 如果抽取出多个命名实体, 依次按照不重叠原则提取⾄tmp列表中
    tmp = []
    for item in predict:
        if not tmp:
            tmp.append(item)
        else:
            if item[1] > tmp[-1][2]:
                tmp.append(item)
    
    # 以原始⽂本为基准, 初始化全'O'的标签列表
    result = ['O'] * len(raw_text)
    for item in tmp:
        # 提取起始索引s, 结束索引e, 实体标签flag
        s, e, flag = item[1], item[2], id2ent[item[3]]
        
        # 如果结束索引在起始索引的右侧, 则可以组装成BI*E的格式
        if e > s:
            result[s] = 'B-{}'.format(flag)
            result[e] = 'E-{}'.format(flag)
            # 中间字符全部设置为I标签
            if e - s > 1:
                for i in range(s + 1, e):
                    result[i] = 'I-{}'.format(flag)
        
        # 如果结束索引==起始索引, 说明是单⼀字符, 设置为S标签
        if e == s:
            result[s] = 'S-{}'.format(flag)
    
    # text: *st恒康重组会停牌多久
    # result: ['B-stock_name', 'I-stock_name', 'I-stock_name', 'I-stock_name', 'E-stock_name', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    return result