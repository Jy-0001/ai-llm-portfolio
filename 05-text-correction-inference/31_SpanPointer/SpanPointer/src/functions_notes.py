# 加载模型 & 放置到GPU中 (单卡/多卡)
def load_model_and_parallel(model, gpu_ids, ckpt_path=None, strict=True):
    '''
        Args:
        model: 已经完成初始化构建的模型
        gpu_ids: 服务器上的GPU列表, 或者是类似于[0, 1, 2, 3], 或者是字符串"-1"表示CPU
        ckpt_path: 模型参数的保存路径, 是可以直接寻址加载的路径
        strict: True或者False, 表示加载模型参数时是否严格遵循key值匹配
        Return:
        model: 加载后的模型
        device: 模型加载到的设备, GPU or CPU
    '''
# 获取模型存放路径的函数
def get_model_path_list(base_dir):
    '''
        Args:
        base_dir: 当前路径, str类型, 例如 /root/server
        Return:
        model_lists: 路径下所有模型的完整地址列表, ['path1', 'path2', ......]
    '''
# SWA滑动平均模型的函数代码
def swa(model, model_dir):
    '''
        Args:
        model: 训练中的模型
        model_dir: 若⼲模型参数的存放路径, str类型, 例如 /root/server
        Return:
        swa_model: 经过SWA处理后的完整参数的模型
    '''