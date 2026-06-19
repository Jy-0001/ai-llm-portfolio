import os                                  # 导入：用于路径拼接/定位文件目录（不改变数据流本质）
import ahocorasick                         # 导入：AC自动机，用来“句子里一次性找多个关键词”
from importlib import import_module
import numpy as np
import torch
import time

UNK, PAD, CLS = '[UNK]', '[PAD]', '[CLS]'

class QuestionClassifier:
    def __init__(self):
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])          # cur_dir:str 当前py文件所在目录
        #　特征词路径
        self.disease_path    = os.path.join(cur_dir, 'dict/disease.txt')       # disease_path:str
        self.food_path       = os.path.join(cur_dir, 'dict/food.txt')          # food_path:str
        self.symptom_path    = os.path.join(cur_dir, 'dict/symptom.txt')       # symptom_path:str
        self.drug_path       = os.path.join(cur_dir, 'dict/drug.txt')          # drug_path:str
        self.department_path = os.path.join(cur_dir, 'dict/department.txt')    # department_path:str
        self.check_path      = os.path.join(cur_dir, 'dict/check.txt')         # check_path:str
        self.producer_path   = os.path.join(cur_dir, 'dict/producer.txt')      # producer_path:str
        self.deny_path       = os.path.join(cur_dir, 'dict/deny.txt')          # deny_path:str（否定词表：不/不能/忌等）
        
        # 加载特征词
        self.disease_words    = [i.strip() for i in open(self.disease_path) if i.strip()]      # disease_words:list[str]
        self.food_words       = [i.strip() for i in open(self.food_path) if i.strip()]         # food_words:list[str]
        self.symptom_words    = [i.strip() for i in open(self.symptom_path) if i.strip()]      # symptom_words:list[str]
        self.drug_words       = [i.strip() for i in open(self.drug_path) if i.strip()]         # drug_words:list[str]
        self.department_words = [i.strip() for i in open(self.department_path) if i.strip()]   # department_words:list[str]
        self.check_words      = [i.strip() for i in open(self.check_path) if i.strip()]        # check_words:list[str]
        self.producer_words   = [i.strip() for i in open(self.producer_path) if i.strip()]     # producer_words:list[str]

        # 构造领域actree
        self.region_words = set(self.disease_words + self.drug_words + self.food_words + self.symptom_words + self.department_words + self.check_words + self.producer_words)  # region_words:set[str] 四类实体词合并去重
        self.region_tree  = self.build_actree(list(self.region_words))         # region_tree:Automaton 用全部实体词构建AC自动机（用于匹配question）
        #构建词典
        self.wdtype_dict  = self.build_wdtype_dict()                           # wdtype_dict:dict[str,list[str]] 实体词 -> 类型列表（可能多标签）
        self.CLS = ['CLS']
        self.pad_size = 40
        # 初始化bert模型，用于意图识别
        self.init_bert()

    def init_bert(self):
        model_name = 'bert_model'
        # x = import_module('bertmodel.' + model_name)
        x = import_module(model_name)
        config = x.Config('red_spider')

        self. model = x.Model(config).to(config.device)

        self.model.load_state_dict(torch.load(config.save_path))

        self.tokenizer = config.tokenizer
        
        # self.symptom_request = ['症状', '表征', '现象', '症候', '表现']          # symptom_request:list[str] “问症状”的触发词
        # self.food_request    = ['饮⻝', '饮⽤', '吃', '⻝', '伙⻝', '膳⻝', '喝', '菜' ,'忌⼝', '补品', '保健品', '⻝谱', '菜谱', '⻝⽤', '⻝物','补品']  # food_request:list[str]
        # self.drug_request    = ['药', '药品', '⽤药', '胶囊', '⼝服液', '炎⽚']    # drug_request:list[str]
        
        print('model init finished...')                                        # 日志：初始化结束（此时已经准备好“词表+自动机+映射字典”）

    def question_class(self, question):
        # 将模型设置为推理模式
        self.model.eval()

        # 进行数据预处理
        tokens = self.tokenizer.tokenize(question)
        tokens = self.CLS + tokens
        mask = []
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        length = len(token_ids)

        # 补齐，阶段，构造mask
        if length < self.pad_size:
            mask = [1] * length + [0] * (self.pad_size - length)
            token_ids += [0] * (self.pad_size - length)
        else:
            mask = [1] * self.pad_size
            token_ids = token_ids[:self.pad_size]

        # 类型封装
        token_ids = torch.LongTensor(token_ids).to('cuda')
        length = torch.LongTensor(length).to('cuda')
        mask = torch.LongTensor(mask).to('cuda')

        # 维度设置
        token_ids = token_ids.unsqueeze(0)
        mask = mask.unsqueeze(0)
        input_ids = (token_ids, length, mask)

        # 直接输入模型进行forward计算，得到最后一层CLS的输出
        output = self.model(input_ids)
        print('output:', output)

        # 通过CLS的输出张量，进行预测
        predict_res = torch.max(output.data, 1)[1].cpu().numpy()

        return predict_res[0]


    def classify(self, question):
        data = {}                                                              # data:dict 最终输出容器（后面塞 args / question_types）

        medical_dict = self.check_medical(question)                             # medical_dict:dict[str,list[str]] 句子中识别到的实体词 -> 类型列表
        if not medical_dict:                                                    # 如果一个实体词都没识别出来
            return {}                                                           # 返回空dict（上层会当成“无法分类”）

        data['args'] = medical_dict                                              # data['args'] 固定装实体识别结果：dict[str,list[str]]

        types = []                                                              # types:list[str] 收集本句出现过的所有实体类型（disease/drug/food/symptom）
        for type_list in medical_dict.values():                                 # type_list:list[str] 逐个取出每个实体的“类型列表”
            types += type_list                                                  # types:list[str] 拉平成一维（可能重复）

        question_type = 'others'                                                     
        question_types = []                                                     # question_types:list[str] 规则分类输出（可多个）

        res_id = self.question_class(question) # 只推理一次
        if res_id == 0 and ('disease' in types):
            question_type = 'disease_drug'
            question_types.append(question_type)

        if res_id == 1 and ('disease' in types):
            question_type = 'disease_food'
            question_types.append(question_type)

        if res_id == 2 and ('disease' in types):
            question_type = 'disease_symptom'
            question_types.append(question_type)

        if res_id == 3 and ('disease' in types):
            question_type = 'disease_cureway'
            question_types.append(question_type)

        if res_id == 4 and ('disease' in types):
            question_type = 'disease_department'
            question_types.append(question_type)

        if res_id == 5 and ('disease' in types):
            question_type = 'disease_check'
            question_types.append(question_type)

        if res_id == 6 and ('disease' in types):
            question_type = 'disease_accompany'
            question_types.append(question_type)

        if res_id == 7 and ('disease' in types):
            question_type = 'disease_producer'
            question_types.append(question_type)

        if res_id == 8 and ('disease' in types):
            question_type = 'disease_category'
            question_types.append(question_type)

        if res_id == 9 and ('disease' in types):
            question_type = 'disease_noteat'
            question_types.append(question_type)


        
        
        if question_type == [] and 'symptom' in types:                         # 兜底：没有触发词分类，但识别到了 symptom 实体
            question_types = ['disease_symptom']                                # question_types:list[str] 直接赋值成默认症状类（注意不是append）

        # 将多个分类结果进行合并处理，组装成一个字典
        data['question_types'] = question_types                                  # data['question_types'] 固定装问句类型list[str]
        
        return data                                                             # 返回结构：{"args":..., "question_types":...}

    def build_wdtype_dict(self):
        word_dict = dict()                                                      # word_dict:dict[str,list[str]] 词 -> 类型列表
        for word in self.region_words:                                          # word:str 遍历所有实体词
            word_dict[word] = []                                                # 初始化该词的类型列表为空 list[str]

            if word in self.disease_words:                                      # 判断该词是否存在于疾病词表
                word_dict[word].append('disease')                               # 类型列表追加 'disease'

            if word in self.drug_words:                                         # 判断该词是否存在于药品词表
                word_dict[word].append('drug')                                  # 类型列表追加 'drug'

            if word in self.food_words:                                         # 判断该词是否存在于食物词表
                word_dict[word].append('food')                                  # 类型列表追加 'food'

            if word in self.symptom_words:                                      # 判断该词是否存在于症状词表
                word_dict[word].append('symptom')                               # 类型列表追加 'symptom'

            if word in self.department_words:
                word_dict[word].append('department')                             # 词 -> department

            if word in self.check_words:
                word_dict[word].append('check')                                  # 词 -> check

            if word in self.producer_words:
                word_dict[word].append('producer')                               # 词 -> producer

        return word_dict                                                        # 返回 dict[str, list[str]]

    def build_actree(self, wordlist):
        actree = ahocorasick.Automaton()                                        # actree:Automaton 创建空AC自动机
        for index, word in enumerate(wordlist):                                 # index:int word:str
            actree.add_word(word, (index, word))                                # 注册关键词word；payload=(index,word)用于匹配时返回
        actree.make_automaton()                                                 # 构建自动机（没有这句 iter 不能高效工作）
        return actree                                                           # 返回 Automaton

    def check_medical(self, question):
        region_words = []                                                       # region_words:list[str] 收集“句子里命中的实体词”（可能含子词）

        for m in self.region_tree.iter(question):                               # m:tuple 形如 (end_index, (idx, word))
            word = m[1][1]                                                      # word:str 取出匹配到的实体词
            region_words.append(word)                                           # region_words:list[str] 追加命中词

        stop_words = []                                                         # stop_words:list[str] 子词列表：用于剔除“词中词”的短词
        for word1 in region_words:                                              # word1:str
            for word2 in region_words:                                          # word2:str
                if word1 in word2 and word1 != word2:                           # 如果 word1 是 word2 的子串（更短）
                    stop_words.append(word1)                                    # 将短词加入 stop_words（待剔除）

        final_words = [w for w in region_words if w not in stop_words]          # final_words:list[str] 剔除子词后的实体词列表
        final_dict  = {w: self.wdtype_dict.get(w) for w in final_words}         # final_dict:dict[str,list[str]] 实体词 -> 类型列表
        
        return final_dict                                                       # 返回 medical_dict 给 classify 用

    # 基于特征词进行问句检测，并进行问句类型的规则分类
    def check_words(self, words, sent):
        for word in words:                                                      # word:str 遍历触发词表
            if word in sent:                                                    # 子串命中
                return True                                                     # 返回 bool
        return False                                                            # 返回 bool
    
if __name__ == '__main__':
    qc = QuestionClassifier()
    while True:
        question = input('input an question:')
        if question == 'q' or question == 'Q':
            break 
        else:
            question = str(question).strip()
            data = qc.classify(question)
            print(data)