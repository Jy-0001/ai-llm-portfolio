import torch
import torch.nn as nn
from transformers import BertForMaskedLM
from base_model import FocalLoss


# 构建macBert4CSC模型类
class Bert4Csc(torch.nn.Module):
    def __init__(self, args, tokenizer):
        super(Bert4Csc, self).__init__()
        self.args = args
        
        # 此处初始化的原始模型是哈工大开源的macBERT
        self.bert = BertForMaskedLM.from_pretrained(args.bert_dir)
        
        # 检测网络对应的是从768向1的映射全连接层
        self.detection = nn.Linear(self.bert.config.hidden_size, 1)
        
        # 二分类的计算函数
        self.sigmoid = nn.Sigmoid()
        
        # tokenizer采用参数模式传入
        self.tokenizer = tokenizer
        
        # 默认配置文件中, 将hyper_params设置为0.2
        self.w = args.hyper_params
    
    # 批次数据的编码函数, 主要为了将原始文本编码成BERT需要的若干重要张量
    def batch_encode(self, batch_data):
        # 最大长度+2, 为[CLS], [SEP]预留空间
        max_len = max([len(x) for x in batch_data]) + 2
        # max_len: 148
        input_ids, token_type_ids, attention_mask = [], [], []
        
        # for text in batch_data:
            # 下面的代码用于当前训练环境, 不报Warning. 但是在转ONNX时报错.
            # inputs = self.tokenizer.encode_plus(text=list(text),
            #                                     max_length=max_len,
            #                                     pad_to_max_length=True,
            #                                     is_split_into_words=True,
            #                                     return_token_type_ids=True,
            #                                     return_attention_mask=True,
            #                                     truncation=True)

            # inputs = self.tokenizer.encode_plus(text=list(text),
            #                                     max_length=max_len,
            #                                     padding='max_length',
            #                                     is_split_into_words=True,
            #                                     return_tensors='pt',
            #                                     truncation=True
            #                                     )

            # 下面的代码用于训练时, 报Warning. 在转ONNX时正确.
            # inputs = self.tokenizer.encode_plus(text=list(text),
            #                                     max_length=max_len,
            #                                     pad_to_max_length=True,
            #                                     is_pretokenized=True,
            #                                     return_token_type_ids=True,
            #                                     return_attention_mask=True,
            #                                     truncation=True)


            # 提取未来BERT模型所需的3个重要张量
            # input_ids.append(inputs['input_ids'])
            # token_type_ids.append(inputs['token_type_ids'])
            # attention_mask.append(inputs['attention_mask'])
       
        # encoded_inputs = self.tokenizer.batch_encode_plus(
        #                                                   batch_data,
        #                                                   max_length=max_len,
        #                                                   padding='max_length',
        #                                                   truncation=True,
        #                                                   return_tensors="pt",
        #                                                   return_token_type_ids=True,
        #                                                   return_attention_mask=True
        #                                                   )

        encoded_inputs = self.tokenizer(
                                        batch_data,
                                        max_length=max_len,
                                        padding='max_length',
                                        truncation=True,
                                        return_tensors="pt",
                                        return_token_type_ids=True,
                                        return_attention_mask=True
                                        )



        # 最后以字典类型返回3个张量
        return {'input_ids': encoded_inputs['input_ids'], 'attention_mask': encoded_inputs['attention_mask'], 'token_type_ids': encoded_inputs['token_type_ids']}

    # macBert4CSC模型的前向计算逻辑
    def forward(self, texts, cor_labels=None, det_labels=None, device=None):
        # 如果传入了正确的标签
        # if cor_labels:
        if cor_labels is not None:
            # 对正确的标签进行batch级别的编码
            # cor_labels: ('中华企业股份怎么操作', '600549.SH资金面得分', ... , '韦尔股份股价走势')
            text_labels = self.batch_encode(cor_labels)['input_ids']
            # text_labels: tensor([[ 101, 1290,  785,  ...,    0,    0,    0],
            #                      [ 101,  127,  121,  ...,    0,    0,    0],
            #                      [ 101,  121,  121,  ...,    0,    0,    0],
            #                       ...,
            #                      [ 101, 3885,  977,  ...,    0,    0,    0],
            #                      [ 101, 6934, 3124,  ...,    0,    0,    0],
            #                      [ 101, 6818,  124,  ...,  126, 2399,  102]])
            # text_labels.shape: [32, 148]
            
            # 对于标签编码等于0的位置(本质上是PAD的位置), 赋值成-100, 未来计算损失时会起到忽略不计的效果
            text_labels[text_labels == 0] = -100
            # text_labels: tensor([[ 101, 1290,  785,  ..., -100, -100, -100],
            #                      [ 101,  127,  121,  ..., -100, -100, -100],
            #                      [ 101,  121,  121,  ..., -100, -100, -100],
            #                       ...,
            #                      [ 101, 3885,  977,  ..., -100, -100, -100],
            #                      [ 101, 6934, 3124,  ..., -100, -100, -100],
            #                      [ 101, 6818,  124,  ...,  126, 2399,  102]])
            text_labels = text_labels.to(device)
        
        # 如果没有传入正确的标签, 则将标签张量text_labels赋值为None
        else:
            text_labels = None

        # 对传入文本进行batch级别的编码
        encoded_text = self.batch_encode(texts)
        # encoded_text: {'input_ids': tensor([[ 101, 1290,  785,  ...,    0,    0,    0],
        #                                     [ 101,  127,  121,  ...,    0,    0,    0],
        #                                     [ 101,  121,  121,  ...,    0,    0,    0],
        #                                      ...,
        #                                     [ 101, 3885,  977,  ...,    0,    0,    0],
        #                                     [ 101, 4507, 3124,  ...,    0,    0,    0],
        #                                     [ 101, 6818,  124,  ...,  126, 2399,  102]]),
        #           'attention_mask': tensor([[1., 1., 1.,  ..., 0., 0., 0.],
        #                                     [1., 1., 1.,  ..., 0., 0., 0.],
        #                                     [1., 1., 1.,  ..., 0., 0., 0.],
        #                                      ...,
        #                                     [1., 1., 1.,  ..., 0., 0., 0.],
        #                                     [1., 1., 1.,  ..., 0., 0., 0.],
        #                                     [1., 1., 1.,  ..., 1., 1., 1.]]),
        #           'token_type_ids': tensor([[0, 0, 0,  ..., 0, 0, 0],
        #                                     [0, 0, 0,  ..., 0, 0, 0],
        #                                     [0, 0, 0,  ..., 0, 0, 0],
        #                                      ...,
        #                                     [0, 0, 0,  ..., 0, 0, 0],
        #                                     [0, 0, 0,  ..., 0, 0, 0],
        #                                     [0, 0, 0,  ..., 0, 0, 0]])}

        # encoded_text作为字典类型, 拥有3种张量
        # 3种keys(): 'input_ids', 'attention_mask', 'token_type_ids'
        for key in encoded_text.keys():
            encoded_text[key] = encoded_text[key].to(device)
        
        # 直接将编码后的张量送入macBERT模型中, 得到输出张量
        bert_outputs = self.bert(**encoded_text,
                                 labels=text_labels,
                                 return_dict=True,
                                 output_hidden_states=True)

        # bert_outputs包含多个输出变量
        # 最后一层隐藏层的输出张量, 送入Detection网络, 得到检错概率
        prob = self.detection(bert_outputs.hidden_states[-1])
        # prob: tensor([[[-0.5638],
        #                [-0.1053],
        #                [-0.2071],
        #                 ...,
        #                [-0.0595],
        #                [-0.4521],
        #                [-0.0822]],
        #
        #               [[-0.2064],
        #                [-0.1139],
        #                [-0.5175],
        #                 ...,
        #                [-0.4933],
        #                [-0.5516],
        #                [-0.3625]],
        #                 ...
        #                [-0.4661]]], device='cuda:0', grad_fn=<AddBackward0>)
        # prob.shape: [32, 148, 1]

        # 如果没有传入正确的标签, predict函数调用, 推理阶段
        if text_labels is None:
            # 最后返回2个张量: Detection网络的检错概率, 和MacBert网络的输出概率分布
            outputs = (prob, bert_outputs.logits)
        # 如果传入了正确的标签, 训练阶段
        else:
            # det_labels: 数据迭代器中, 已经手动构建好的错误字符标签的one-hot格式张量
            det_labels = det_labels.to(device)
            # det_labels:  tensor([[0, 0, 0,  ..., 0, 0, 0],
            #                      [0, 0, 0,  ..., 0, 0, 0],
            #                      [0, 0, 0,  ..., 0, 0, 0],
            #                       ...,
            #                      [0, 0, 0,  ..., 0, 0, 0],
            #                      [0, 0, 0,  ..., 0, 0, 0],
            #                      [0, 0, 0,  ..., 0, 0, 0]]
            # det_labels.shape: [32, 148]

            # 设置损失函数的计算规则为FocalLoss的二分类模式sigmoid
            det_loss_fct = FocalLoss(num_labels=None, activation_type='sigmoid').cuda()
            # det_loss_fct: FocalLoss()

            # pad部分不计算损失, 只把mask == 1的位置计算有效损失
            active_loss = encoded_text['attention_mask'].view(-1, prob.shape[1]) == 1
            # active_loss:  tensor([[ True,  True,  True,  ..., False, False, False],
            #                       [ True,  True,  True,  ..., False, False, False],
            #                       [ True,  True,  True,  ..., False, False, False],
            #                        ...,
            #                       [ True,  True,  True,  ..., False, False, False],
            #                       [ True,  True,  True,  ..., False, False, False],
            #                       [ True,  True,  True,  ..., False, False, False]]
            # active_loss.shape: [32, 148]

            # Detection网络的检错概率, 进行mask掩码后, 作为有效的检错概率分布张量active_probs
            active_probs = prob.view(-1, prob.shape[1])[active_loss]
            # active_probs:  tensor([-2.5114e-01, -2.7376e-01, -4.0016e-01, -1.8689e-01, -1.8178e-01,
            #                        -8.0171e-02, -4.0444e-02, -2.1395e-01, -7.5420e-01, -3.6822e-01,
            #                        -1.9996e-01, -3.1433e-01, -4.6916e-01, -5.5020e-01, -9.0304e-01,
            #                        -8.7629e-01, -9.6466e-01, -8.1219e-01, -5.8654e-01, -1.1852e+00,
            #                        -1.0156e+00, -6.3518e-03, -6.1101e-01, -4.2934e-01, -2.1000e-01,
            #                        ......
            #                        -4.1340e-01, -9.3023e-01, -4.0680e-01]
            # active_probs.shape: [853]

            # 检错标签, 进行mask掩码后, 作为有效的检错分布标签
            active_labels = det_labels[active_loss]
            # active_labels:  tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #                         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #                         ......
            #                         0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
            # active_labels.shape: [853]

            # Detection网络的有效输出概率active_probs, 和有效检错标签active_labels, 进行FocalLoss计算
            det_loss = det_loss_fct(active_probs, active_labels.float())
            # det_loss:  tensor(0.0692, device='cuda:0', grad_fn=<MeanBackward0>)

            # 按照macBert4CSC计算公式, macBERT的输出损失, 和检错网络的FocalLoss损失, 进行加权求和.
            loss = self.w * bert_outputs.loss + (1 - self.w) * det_loss
            # loss:  tensor(0.5683, device='cuda:0', grad_fn=<AddBackward0>)

            # 最后返回3个张量: 网络的总损失loss, Detection网络的检错概率, macBERT网络的输出概率分布
            outputs = (loss, self.sigmoid(prob).squeeze(-1), bert_outputs.logits)
        
        return outputs

    # 预测函数, 本质上执行的是推理过程
    def predict(self, texts, device):
        # 首先对原始文本进行编码, 生成输入张量inputs
        # texts: ['国星光电']
        inputs = self.batch_encode(texts)

        with torch.no_grad():
            # 得到Detection网络的检错输出, 以及macBERT网络的纠错输出
            outputs = self.forward(texts, device=device)
            
            # 在本函数中, outputs[1]代表macBERT网络的输出概率分布张量, 而不是Detection的检错张量
            # 在forward函数中, if分支代表当前predict的真实调用, else分支代表训练阶段的调用
            # outputs[1]: tensor([[[ -5.8146,  -5.7773,  -4.7123,  ...,  -2.6241,  -5.5271,  -5.3183],
            #                      [-11.9535, -10.4815, -11.1018,  ...,  -7.1400, -11.4753,  -4.2355],
            #                      [-13.2651, -10.6365,  -9.6797,  ..., -10.1722, -13.0885,  -5.8356],
            #                      [-10.8641,  -9.0556,  -4.7642,  ...,  -7.3156,  -8.2596,  -3.3446],
            #                      [-10.2837,  -6.2099,  -7.3063,  ...,  -7.7704, -12.3898,  -6.5550],
            #                      [ -5.6941,  -5.7307,  -4.7791,  ...,  -3.2906,  -4.4010,  -4.7431]]]
            
            y_hat = torch.argmax(outputs[1], dim=-1)
            # y_hat: tensor([[ 102, 1744, 3215, 1045, 4510,  102]])
            # y_hat.shape: [1, 6]
            
            # 对attention_mask求和得出有效文本长度, -1是为了删除最后面的[SEP]
            # inputs['attention_mask']: tensor([[1., 1., 1., 1., 1., 1.]])
            expand_text_lens = torch.sum(inputs['attention_mask'], dim=-1) - 1
            # expand_text_lens: [5.]

        # 初始化batch解码的结果列表
        res = []
        # 每一条文本的有效长度不同, 因此一一对应的处理
        for t_len, _y_hat in zip(expand_text_lens, y_hat):
            t_len = t_len.long()
            # t_len: 5
            # _y_hat: tensor([ 102, 1744, 3215, 1045, 4510,  102])
            # self.tokenizer.decode(_y_hat[1: t_len]).replace(' ', ''): 国星光电
            
            # 直接调用tokenizer.decode()对检错网络按照贪心算法argmax()来进行解码, 并将空格全部删除
            res.append(self.tokenizer.decode(_y_hat[1: t_len]).replace(' ', ''))
        
        return res

