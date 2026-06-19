'''配置函数的实现'''

'''标签/索引字典'''
# 设定label_to_id映射字典，采用BIEO标注模式，外加3个特殊字符<pad>, <start>, <eos>
l2i_dic = {'O': 0, u'B-sym': 1, u'B-dis': 2, u'I-sym': 3, u'I-dis': 4, u'E-sym': 5, u'E-dis': 6, '<pad>': 7, '<start>': 8, '<eos>': 9} #sym代表symptoms症状，dis代表disease病名

# 上面的逆字典：id_to_label
i2l_dic = {0: 'O', 1: u'B-sym', 2: u'B-dis', 3: u'I-sym', 4: u'I-dis', 5: u'E-sym', 6: u'E-dis', 7: '<pad>', 8: '<start>', 9: '<eos>'}

'''训练集, 测试集, 词表'''
train_file = './data/train.txt'
dev_file = './data/test.txt'
vocab_file = './data/vocab.txt'

save_model_path = './saved_model/idcnn_crf.pt'

model_path = './saved_model/idcnn_crf.pt'

'''设置超参数'''
max_length = 256 # 每条序列的最大长度，超过不要，短了补pad
batch_size = 32
epochs = 35 # 因为不是预训练模型，epoch可以多跑几个
lr = 1e-5 # 因为没有用预训练模型，学习率采用千分之一
tagset_size = len(l2i_dic) #标签集合（模型输出所有可能标签）
dropout = 0.4 # 有卷积操作一般会设置dropout
use_cuda = True
