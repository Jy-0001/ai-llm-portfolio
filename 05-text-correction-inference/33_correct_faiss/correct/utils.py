'''==========================⽂本距离的衡量=========================='''
import os
import numpy as np
from subprocess import Popen, PIPE, STDOUT
import argparse

# 函数按照最⼩编辑距离的算法来进⾏计算, 通过字符的添加, 修改, 删除来匹配⽂本
def edit_distance(string_a, string_b, name='Levenshtein'):
    size_x = len(string_a) + 1
    size_y = len(string_b) + 1

    # 初始化全0的编辑距离矩阵
    matrix = np.zeros((size_x, size_y), dtype=int)

    # 初始化边界条件, string_b为空的情况
    for x in range(size_x):
        matrix[x, 0] = x

    # 初始化边界条件, string_a为空的情况
    for y in range(size_y):
        matrix[0, y] = y

    # 依次进⾏双重循环遍历, 按照3种情况, 插⼊, 修改, 删除, 其中距离最⼩的来更新矩阵参数
    for x in range(1, size_x):
        for y in range(1, size_y):
            # 前⼀个⼦串匹配
            if string_a[x - 1] == string_b[y - 1]:
                matrix[x, y] = min(matrix[x-1, y] + 1, 
                                   matrix[x-1, y-1], 
                                   matrix[x, y-1] + 1)

            # 前⼀个⼦串不匹配
            else:
                # 采⽤最经典的算法
                if name == 'Levenshtein':
                    matrix[x, y] = min(matrix[x-1, y] + 1, 
                                       matrix[x-1, y-1] + 1, 
                                       matrix[x, y-1] + 1)
                    
    return matrix[size_x - 1, size_y - 1]

# 将输⼊⽂本进⾏tokenizer的操作后, 返回能输⼊模型的张量
def generate_input(text, tokenizer):
    input_ids, token_type_ids, attention_mask = [], [], []
    # encode_dict = tokenizer._encode_plus(text=list(text), 
    #                                     pad_to_max_length=True, 
    #                                     is_pretokenized=True, 
    #                                     return_token_type_ids=True, 
    #                                     return_attention_mask=True, 
    #                                     return_tensors='pt').to('cuda')
    encode_dict = tokenizer(text, 
                            padding="max_length",
                            truncation=True,
                            max_length=128,
                            return_tensors="pt",
                            ).to('cuda')
    
    
    return encode_dict

# 进⾏指针span模式的NER解码函数
def span_decode(start_logits, end_logits, raw_text, id2ent):
    predict=[]
    # 起始指针, 结束指针的预测位置
    start_pred = np.argmax(start_logits, -1)
    end_pred = np.argmax(end_logits, -1)

    # 双重循环遍历起始指针, 结束指针, 利⽤pointer span的模式进⾏实体抽取
    for i, s_type in enumerate(start_pred):
        if s_type == 0:
            continue
        for j, e_type in enumerate(end_pred[i:]):
            # 起始指针, 结束指针, 两者类型相同, 说明提取出了⼀个有效实体
            if s_type == e_type:
                # 通过原始⽂本将实体截取出来
                tmp_ent = raw_text[i: i + j + 1]
                # 将实体, start, end, 实体类型, 总共4个信息打包添加进预测结果列表中
                predict.append((''.join(tmp_ent), i, i + j, s_type))
                # NER策略: ⼀个start位置, 只提取⼀个实体, 不进⾏同位置的嵌套实体的抽取
                break

    # 双重for循环结束后, 提取预测出的实体
    tmp = []
    for item in predict:
        # 结果列表predict中的第⼀个实体, ⼀定添加进tmp中
        if not tmp:
            tmp.append(item)
        else:
            # 后续添加的实体, 必须保证起始位置⼤于上⼀个实体的结束位置, 也就是⾮嵌套实体.
            if item[1] > tmp[-1][2]:
                tmp.append(item)

    # 真实字符串ent才是我们想要的实体结果
    res = []
    for ent, _, _, _ in tmp:
        res.append(ent)

    return res








