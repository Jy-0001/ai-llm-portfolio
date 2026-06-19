'''数据集类代码实现'''
'''接口设计'''
class NERDataset(Dataset):
    # 初始化函数
    def __init__(self,):
        pass
    # 测量数据个数的函数
    def __len__(self,):
        pass
    # 按索引获取数据的函数
    def __getitem__(self,):
        pass
    # 根据原始⽂本和索引数据, 构造NER的标签列表数据
    def get_bieo_data(self,):
        pass
    # 个性化数据格式处理的函数
    def collate_fn(self,):
        pass

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
