import argparse
from torch.utils.data import Dataset

class Args:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()
        return parser
    
    @staticmethod
    def initialize(parser:argparse.ArgumentParser):
        parser.add_argument('--raw_data_dir', default='./data', help='数据集的存放路径')
        parser.add_argument('--output_dir', default='./out/', help='训练好的模型输出路径')
        parser.add_argument('--bert_dir', default='./src/bert-base-chinese', help='可以⽀持ernie, roberta-wwm, bert, macbert')
        parser.add_argument('--task_type', default='span', help='NER任务采⽤的模型类别, crf/span')
        parser.add_argument('--loss_type', default='ls_ce', help='损失函数的类型, crf/span')
        # other args
        parser.add_argument('--seed', type=int, default=2023, help='随机种⼦, random seed')
        parser.add_argument('--gpu_ids', type=str, default=['0'], help='GPU的id信息, "-1"代表cpu, "0,1,..."代表GPU')
        parser.add_argument('--mode', type=str, default='train', help='当前模式, 训练或推理')
        # train args
        parser.add_argument('--train_epochs', default=10, type=int, help='训练模型的轮次数')
        parser.add_argument('--dropout_prob', default=0.1, type=float, help='dropout⽐例')
        parser.add_argument('--lr', default=2e-5, type=float, help='学习率, 针对BERT系列')
        parser.add_argument('--other_lr', default=2e-3, type=float, help='学习率, 针对BERT外的模块')
        parser.add_argument('--max_grad_norm', default=1.0, type=float, help='最⼤梯度裁剪值')
        parser.add_argument('--warmup_proportion', default=0.1, type=float)
        parser.add_argument('--weight_decay', default=0.01, type=float)
        parser.add_argument('--adam_epsilon', default=1e-8, type=float)
        parser.add_argument('--train_batch_size', default=64, type=int)
        parser.add_argument('--test_file', default='')

        return parser
    
    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args()


'''明确类中每一个函数的具体接口'''
class NERDataset(Dataset):
    def __init__(self, train_feature, config, ent2id):
        '''
            Args:
                train_feature:数据集
                config:配置信息，以parser命令参数配置的模式传入
                ent2id:实体类型的映射字典，本项目中只有stock_name一种实体类型
        '''
    def __len__(self):
        '''
            Args:
                None:无需输入参数
            Return:
                长度，一个整形数字
        '''
    def __getitem__(self, index):
        '''
            Args: 
                index: 下标, ⼀个整型数字
            Return:
                self.data[index]
        '''
    def get_bieo_data(self, text, data):
        '''
            Args: 
                text: str类型, 为⼀段⽂本
                data: list类型, 包含对应于text的索引数据
            Return:
                labels: list类型, 为实例化具体标签值后, 对应于text的标签列表
        '''
    def collate_fn(self, batch_data):
        '''
            Args: 
                batch_data: list类型, 每个元素为⼀个dict类型, 包含'text', 'stock_name'两个key值
            Return:
                dict类型字典, key = ['input_ids', 'token_type_ids','attention_mask', 'raw_text','start_ids','end_ids','bieo_labels']
        '''





