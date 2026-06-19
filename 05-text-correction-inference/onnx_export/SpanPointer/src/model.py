# 模型类代码
'''导入需要的库'''
import os
import math
import torch
import torch.nn as nn
from itertools import repeat
from transformers import BertModel
# from src.evaluate import span_decode
from .evaluate import span_decode
import pdb

# 标签平滑的交叉熵类代码
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        c = output.size()[-1]
        # ⾸先在最后⼀个维度上正常执⾏log_softmax操作
        log_pred = torch.log_softmax(output, dim=-1)

        # 如果采用求和方案
        if self.reduction == 'sum':
            loss = -log_pred.sum()

        # 如果采用均值方案
        else:
            loss = -log_pred.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()

        # 按照标签平滑的计算公式来计算损失值
        return loss * self.eps / c + (1 - self.eps) * torch.nn.functional.nll_loss(log_pred, target, reduction=self.reduction, ignore_index=self.ignore_index)

'''基础类的模型代码, 仅完成预训练模型的继承和参数模块的初始化'''
class BaseModel(nn.Module):
    def __init__(self, bert_dir, dropout_prob):
        super(BaseModel, self).__init__()
        config_path = os.path.join(bert_dir, 'config.json')
        assert os.path.exists(bert_dir) and os.path.exists(config_path), 'pretrained bert file does not exist'
        
        # 如果继承BERT, RoBERTa, macBERT, ERNIE等模型
        self.bert_module = BertModel.from_pretrained(bert_dir, output_hidden_states=True, hidden_dropout_prob=dropout_prob)
        self.bert_config = self.bert_module.config
    
    @staticmethod
    def _init_weights(blocks, **kwargs):
        # 参数初始化, 将Linear/Embedding/LayerNorm与Bert进⾏⼀样的初始化
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0, std=kwargs.pop('initializer_range', 0.02))
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

'''Span Pointer模型, 采⽤BERT + Span Pointer的架构'''
class SpanModel(BaseModel):
    def __init__(self, bert_dir, num_tags, config, dropout_prob=0.1, loss_type='ce', **kwargs):
        # loss_type: 损失函数的类型, 有三种选择, ['ce', 'ls_ce', 'focal']
        super(SpanModel, self).__init__(bert_dir, dropout_prob=dropout_prob)
        self.config = config
        # 传参进来的参数hidden_size = 768
        out_dims = self.bert_config.hidden_size
        # kwargs.pop()是将配置⽂件中的字典信息删除, 并返回配置值; 如果不存在, 则返回第⼆个参数指定的默认值
        mid_linear_dims = kwargs.pop('mid_linear_dims', 128)
        # 传参进来的参数num_tags = len(ent2id) + 1 = 2
        self.num_tags = num_tags

        # 这⾥out_dims = 768, mid_linear_dims = 128
        self.mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims), 
            nn.ReLU(), 
            nn.Dropout(dropout_prob)
        )

        # 这⾥的两个全连接层是(128, 2)的映射
        self.start_fc = nn.Linear(mid_linear_dims, num_tags)
        self.end_fc = nn.Linear(mid_linear_dims, num_tags)

        reduction = 'none'
        # 如果采⽤交叉熵损失函数
        if loss_type == 'ce':
            self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        # 如果采⽤Label Smoothing交叉熵损失函数, 本项⽬默认采⽤的损失函数ls_ce
        elif loss_type == 'ls_ce':
            self.criterion = LabelSmoothingCrossEntropy(reduction=reduction)
        # 如果采⽤Focal Loss损失函数
        else:
            self.criterion = FocalLoss(reduction=reduction)

        # 初始化损失值权重
        self.loss_weight = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.loss_weight.data.fill_(-0.5)
        
        # 设定参与参数初始化的模型模块, 并执⾏初始化
        init_blocks = [self.mid_linear, self.start_fc, self.end_fc]
        self._init_weights(init_blocks)

    def forward(self, input_ids, attention_mask, token_type_ids, start_ids=None, end_ids=None, pseudos=None, labels=None):
        # 1: 第1步数据先经历BERT的处理, 得到输出张量
        bert_outputs = self.bert_module(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # 1.1: 采⽤第⼀个位置的最后⼀层全部输出, 不⽤[CLS]的张量
        seq_out = bert_outputs[0]
        # seq_out: [64, 28, 768]
        
        # 2: 第2步将BERT的输出张量送⼊全连接层, 映射为[batch_size, seq_len, 128]的输出张量
        seq_out = self.mid_linear(seq_out)
        # seq_out: [64, 28, 128]

        # 3: 第3步继续将128维度的张量送⼊START, END两个映射层, 得到[batch_size, seq_len, 2]的逻辑张量
        start_logits = self.start_fc(seq_out)
        end_logits = self.end_fc(seq_out)
        # start_logits: [64, 28, 2]
        # end_logits: [64, 28, 2]
        
        # 构造返回元组信息
        out = (start_logits, end_logits)

        if start_ids is not None and end_ids is not None and self.training:
            # 将数据维度变成[-1, 2]
            # print('222')
            start_logits = start_logits.view(-1, self.num_tags)
            end_logits = end_logits.view(-1, self.num_tags)
            # start_logits: [1792, 2]
            # end_logits: [1792, 2]
            # 去掉padding部分的标签, 计算真实loss
            # attention_mask.shape: torch.Size([64, 28])
            active_loss = attention_mask.view(-1) == 1
            # len(active_loss) = 1792 = 64 * 28
            # active_loss.sum() = 717, 说明有717个True, 剩余的都等于False
            
            # active_loss: tensor([ True, True, True, ..., False, False, False])
            # start_logits: tensor([[-0.4023, -0.1181],
                                #  [-0.3951, -0.2400],
                                #  [-0.5110, -0.0372],
                                #  ...,
                                #  [-0.1658, -0.0045],
                                #  [ 0.0818, 0.1422],
                                #  [-0.2181, -0.1288]]
            active_start_logits = start_logits[active_loss]
            # active_start_logits: tensor([[-0.4023, -0.1181],
            # [-0.3951, -0.2400],
            # [-0.5110, -0.0372],
            # ...,
            # [-0.2263, -0.0277],
            # [ 0.0203, 0.1713],
            # [-0.3817, 0.1703]]
            # 此处形状第⼀维变成717, 是因为active_loss有717个位置等于True, 其他⾏被过滤掉了
            # active_start_logits.shape: [717, 2]
            
            active_end_logits = end_logits[active_loss]
            # start_ids.shape: [64, 28]
            # start_ids: tensor([[0, 1, 0, ..., 0, 0, 0],
            # [0, 1, 0, ..., 0, 0, 0],
            # [0, 1, 0, ..., 0, 0, 0],
            # ...,
            # [0, 0, 0, ..., 0, 0, 0],
            # [0, 1, 0, ..., 0, 0, 0],
            # [0, 1, 0, ..., 0, 0, 0]]

            active_start_labels = start_ids.view(-1)[active_loss]
            # active_start_labels.shape: torch.Size([717])
            active_end_labels = end_ids.view(-1)[active_loss]
            # 这是本项⽬真实执⾏的损失计算部分
            start_loss = self.criterion(active_start_logits, active_start_labels).mean()
            end_loss = self.criterion(active_end_logits, active_end_labels).mean()
            # start_loss: tensor(0.7804)
            # end_loss: tensor(0.6838)
            # 损失函数分两部分, 最后将返回元组信息构造好
            loss = start_loss + end_loss
            out = (loss, ) + out
        
        return out
    
def build_model(task_type, bert_dir, config, **kwargs):
    if task_type == 'span':
        model = SpanModel(
            bert_dir=bert_dir, 
            num_tags=kwargs.pop('num_tags'), 
            config=config, 
            dropout_prob=kwargs.pop('dropout_prob', 0.1), 
            loss_type=kwargs.pop('loss_type', 'ce')
        )
    return model



