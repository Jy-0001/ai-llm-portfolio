'''问题分类子任务QuestionClassifier'''
import os                                      # 路径/文件定位（本身不产生业务数据）
import ahocorasick                             # AC 自动机：多关键词匹配（核心用于实体识别）
import time                                    # 这里没用到（不影响数据流）

class QuestionClassifier:
    def __init__(self):
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])          # cur_dir:str 当前文件所在目录

        # ===== 1) 词表路径：全部都是 str =====
        self.disease_path    = os.path.join(cur_dir, 'dict/disease.txt')       # disease_path:str
        self.department_path = os.path.join(cur_dir, 'dict/department.txt')    # department_path:str
        self.check_path      = os.path.join(cur_dir, 'dict/check.txt')         # check_path:str
        self.drug_path       = os.path.join(cur_dir, 'dict/drug.txt')          # drug_path:str
        self.food_path       = os.path.join(cur_dir, 'dict/food.txt')          # food_path:str
        self.symptom_path    = os.path.join(cur_dir, 'dict/symptom.txt')       # symptom_path:str
        self.producer_path   = os.path.join(cur_dir, 'dict/producer.txt')      # producer_path:str
        self.deny_path       = os.path.join(cur_dir, 'dict/deny.txt')          # deny_path:str（否定词表：不/不能/忌等）

        # ===== 2) 加载特征词：文件 -> list[str] =====
        self.disease_words    = [i.strip() for i in open(self.disease_path) if i.strip()]      # disease_words:list[str]
        self.drug_words       = [i.strip() for i in open(self.drug_path) if i.strip()]         # drug_words:list[str]
        self.food_words       = [i.strip() for i in open(self.food_path) if i.strip()]         # food_words:list[str]
        self.symptom_words    = [i.strip() for i in open(self.symptom_path) if i.strip()]      # symptom_words:list[str]
        self.department_words = [i.strip() for i in open(self.department_path) if i.strip()]   # department_words:list[str]
        self.check_words      = [i.strip() for i in open(self.check_path) if i.strip()]        # check_words:list[str]
        self.producer_words   = [i.strip() for i in open(self.producer_path) if i.strip()]     # producer_words:list[str]

        # ===== 3) 合并“领域实体词”：list -> set =====
        self.region_words = set(                                                                # region_words:set[str]（去重后的所有实体词）
            self.disease_words + self.drug_words + self.food_words + self.symptom_words
            + self.department_words + self.check_words + self.producer_words
        )

        # ===== 4) 否定词：list[str]（用于 food 的 “能/不能吃” 分支）=====
        self.deny_words = [i.strip() for i in open(self.deny_path) if i.strip()]                # deny_words:list[str]

        # ===== 5) AC 自动机：set[str] -> Automaton =====
        self.region_tree = self.build_actree(list(self.region_words))                            # region_tree:Automaton（实体识别用）

        # ===== 6) “实体词 -> 类型列表”字典：set[str] -> dict[str, list[str]] =====
        self.wdtype_dict = self.build_wdtype_dict()                                              # wdtype_dict:dict[str,list[str]]

        # ===== 7) 触发词表：list[str]（用于规则分类输出 question_types）=====
        self.symptom_request   = ['症状', '表征', '现象', '症候', '表现', '难受', '不舒服', '征兆', '反应', '状况', '不对劲', '感觉']  # list[str]
        self.food_request      = ['饮食', '饮用', '吃', '食', '伙食', '膳食', '喝', '菜', '忌口', '补品', '保健品', '食谱', '菜谱', '食用', '食物', '能吃吗', '可以吃吗', '吃点啥', '食疗', '调理', '水果', '进补']  # list[str]
        self.drug_request      = ['药', '药品', '用药', '胶囊', '口服液', '炎片', '药方', '药剂', '吃药', '什么药', '处方', '偏方', '抗生素', '药片']  # list[str]
        self.cause_request     = ['原因', '成因', '为什么', '怎么会', '怎样才', '咋样才', '怎样会', '如何会', '为啥', '为何', '如何才会', '怎么才会', '会导致', '会造成', '引起', '诱发', '病因']  # list[str]
        self.acompany_request  = ['并发症', '并发', '一起发生', '一并发生', '一起出现', '一并出现', '一同发生', '一同出现', '伴随发生', '伴随', '共现', '跟着', '引发', '关联', '还会得什么']  # list[str]
        self.prevent_request   = ['预防', '防范', '抵制', '抵御', '防止', '躲避', '逃避', '避开', '免得', '逃开', '避开', '避掉', '躲开', '躲掉', '绕开', '怎样才能不', '怎么才能不', '咋样才能不', '咋才能不', '如何才能不', '怎样才不', '怎么才不', '咋样才不', '咋才不', '如何才不', '怎样才可以不', '怎么才可以不', '咋样才可以不', '咋才可以不', '如何可以不', '怎样才可不', '怎么才可不', '咋样才可不', '咋才可不', '如何可不', '怎么防', '怎么避', '注意', '防护']  # list[str]
        self.lasttime_request  = ['周期', '痊愈期', '康复', '恢复', '缓解', '摆脱', '控制', '稳定', '持续', '最佳', '何时', '多久', '多长时间', '多少时间', '几天', '几年', '多少天', '多少小时', '几个小时', '多少年', '几周', '几个月', '多快', '多久见效', '多久能好', '时长', '多久断根']  # list[str]
        self.cureway_request   = ['怎么治疗', '如何医治', '怎么医治', '怎么治', '怎么医', '如何治', '医治方式', '疗法', '咋治', '怎么办', '咋办', '手术', '打针', '输液', '静脉滴注', '挂水', '开刀', '物理治疗', '保守治疗']  # list[str]
        self.cureprob_request  = ['多大概率能治好', '多大几率能治好', '治好希望大么', '几率', '几成', '比例', '可能性', '能治', '可治', '可以治', '可以医', '能不能活', '生命周期', '活多久', '希望', '救不救得了', '几成把握', '彻底治好', '根治']  # list[str]
        self.check_request     = ['检查', '检查项目', '查出', '测出', '试出', '化验', '抽血', '拍片', 'B超', 'CT', '影像', '核磁', '指标', '化验单']  # list[str]
        self.cure_request      = ['治疗什么', '治啥', '治疗措施', '策略', '辅助治疗', '治疗', '治疗啥', '医治啥', '治愈啥', '主治啥', '主治什么', '有什么用', '有何用', '用处', '用途', '有什么好处', '有什么益处', '有何益处', '用来', '用来做啥', '用来作甚', '需要', '要']  # list[str]
        self.easyget_request   = ['易感人群', '容易感染', '易发人群', '什么人', '哪些人', '感染', '染上', '得上']  # list[str]
        self.belong_request    = ['属于什么科', '属于', '什么科', '科室', '哪个科', '挂什么号', '看什么门诊', '挂哪个号', '找哪个医生']  # list[str]

        print('model init finished...')                                          # 日志：初始化完成（此时准备好所有识别/分类所需结构）

    # ======================= 分类主函数：question(str) -> data(dict) =======================
    def classify(self, question):
        data = {}                                                               # data:dict 最终输出容器

        medical_dict = self.check_medical(question)                              # medical_dict:dict[str,list[str]] 实体识别结果（词->类型列表）
        if not medical_dict:                                                     # 若没有识别到任何实体词
            return {}                                                            # 返回空 dict（上层认为无法分类）
        data['args'] = medical_dict                                              # data['args']=dict[str,list[str]] 固定字段：实体识别输出

        types = []                                                               # types:list[str] 本句出现过的所有实体类型（拉平）
        for type_list in medical_dict.values():                                  # type_list:list[str] 每个实体词的类型列表
            types += type_list                                                   # types:list[str] 追加/拼接（可能重复）

        question_type = 'others'                                                 # question_type:str 临时变量（最终其实由 question_types 决定）
        question_types = []                                                      # question_types:list[str] 最终分类标签（可多个）

        # ===== 症状：疾病 -> 症状 =====
        if self.check_word_in_sentence(self.symptom_request, question) and ('disease' in types):  # 触发词命中 + 存在 disease 实体
            question_type = 'disease_symptom'                                    # str：问“某疾病有哪些症状”
            question_types.append(question_type)                                 # list[str] += 1

        # ===== 症状：症状 -> 疾病 =====
        if self.check_word_in_sentence(self.symptom_request, question) and ('symptom' in types):  # 触发词命中 + 存在 symptom 实体
            question_type = 'symptom_disease'                                    # str：问“某症状可能是什么病”
            question_types.append(question_type)                                 # list[str] += 1

        # ===== 病因：疾病 -> 原因 =====
        if self.check_word_in_sentence(self.cause_request, question) and ('disease' in types):    # 触发词命中 + 存在 disease
            question_type = 'disease_cause'                                      # str：问“某疾病的原因/诱因”
            question_types.append(question_type)                                 # list[str] += 1

        # ===== 并发症：疾病 -> 并发症 =====
        if self.check_word_in_sentence(self.acompany_request, question) and ('disease' in types): # 触发词命中 + disease
            question_type = 'disease_accompany'                                   # str：问“某疾病会引发哪些并发症”
            question_types.append(question_type)                                 # list[str] += 1

        # ===== 饮食推荐：疾病 -> 能吃/不能吃（根据否定词） =====
        if self.check_word_in_sentence(self.food_request, question) and ('disease' in types):     # 触发词命中 + disease
            deny_status = self.check_word_in_sentence(self.deny_words, question)                  # deny_status:bool 句子里是否出现否定词
            if deny_status:                                                    # 如果句子含否定（如 不能/忌）
                question_type = 'disease_not_food'                             # str：问“某病不能吃什么”
            else:                                                               # 否定词没出现
                question_type = 'disease_do_food'                               # str：问“某病能吃什么/推荐什么”
            question_types.append(question_type)                                # list[str] += 1（注意：这条一定会append一个）

        # ===== 已知食物 -> 查疾病（同样区分能/不能）=====
        if self.check_word_in_sentence(self.food_request + self.cure_request, question) and 'food' in types:  # 触发词命中 + 句子里识别到 food 实体
            deny_status = self.check_word_in_sentence(self.deny_words, question)                  # deny_status:bool 同上
            if deny_status:
                question_type = 'food_not_disease'                              # str：问“某食物不适合哪些病”
            else:
                question_type = 'food_do_disease'                               # str：问“某食物适合哪些病/对哪些病有用”
            question_types.append(question_type)                                # list[str] += 1

        # ===== 药品推荐：疾病 -> 用药 =====
        if self.check_word_in_sentence(self.drug_request, question) and ('disease' in types):     # 触发词命中 + disease
            question_type = 'disease_drug'                                      # str：问“某病用什么药”
            question_types.append(question_type)                                # list[str] += 1

        # ===== 药品 -> 查疾病（药治啥）=====
        if self.check_word_in_sentence(self.drug_request, question) and ('drug' in types):       # 触发词命中 + drug 实体
            question_type = 'drug_disease'                                      # str：问“某药治什么病”
            question_types.append(question_type)                                # list[str] += 1

        # ===== 疾病 -> 检查项目 =====
        if self.check_word_in_sentence(self.check_request, question) and 'disease' in types:     # 触发词命中 + disease
            question_type = 'disease_check'                                     # str：问“某病要做哪些检查”
            question_types.append(question_type)                                # list[str] += 1

        # ===== 已知检查项目 -> 查疾病 =====
        if self.check_word_in_sentence(self.check_request + self.cure_request, question) and 'check' in types: # 触发词命中 + check 实体
            question_type = 'check_disease'                                     # str：问“某检查提示哪些病”
            question_types.append(question_type)                                # list[str] += 1

        # ===== 预防：疾病 -> 预防措施 =====
        if self.check_word_in_sentence(self.prevent_request, question) and 'disease' in types:   # 触发词命中 + disease
            question_type = 'disease_prevent'                                   # str：问“某病怎么预防”
            question_types.append(question_type)                                # list[str] += 1

        # ===== 周期：疾病 -> 多久能好 =====
        if self.check_word_in_sentence(self.lasttime_request, question) and 'disease' in types:  # 触发词命中 + disease
            question_type = 'disease_lasttime'                                  # str：问“某病持续多久/多久好”
            question_types.append(question_type)                                # list[str] += 1

        # ===== 治疗方式：疾病 -> 怎么治 =====
        if self.check_word_in_sentence(self.cureway_request, question) and 'disease' in types:   # 触发词命中 + disease
            question_type = 'disease_cureway'                                   # str：问“某病怎么治疗”
            question_types.append(question_type)                                # list[str] += 1

        # ===== 治愈概率：疾病 -> 能不能治好 =====
        if self.check_word_in_sentence(self.cureprob_request, question) and 'disease' in types:  # 触发词命中 + disease
            question_type = 'disease_cureprob'                                  # str：问“某病治愈概率”
            question_types.append(question_type)                                # list[str] += 1

        # ===== 易感人群：疾病 -> 什么人容易得 =====
        if self.check_word_in_sentence(self.easyget_request, question) and 'disease' in types:   # 触发词命中 + disease
            question_type = 'disease_easyget'                                   # str：问“某病易感人群”
            question_types.append(question_type)                                # list[str] += 1

        # ===== 科室 =====
        if self.check_word_in_sentence(self.belong_request, question) and 'disease' in types:   # 触发词命中 + disease
            question_type = 'disease_belong'                                   # str：问“某病易感人群”
            question_types.append(question_type)                                # list[str] += 1

        # ===== 兜底1：没命中任何规则，但识别到了 symptom 实体 =====
        if question_types == [] and 'symptom' in types:                          # 如果 question_types 还是空，并且句子里有 symptom 实体
            question_types = ['disease_symptom']                                 # list[str]：默认当“问症状”（注意：直接赋值）

        # ===== 兜底2：没命中任何规则，但识别到了 disease 实体 =====
        if question_types == [] and 'disease' in types:                          # 如果 question_types 还是空，并且句子里有 disease 实体
            question_types = ['disease_desc']                                    # list[str]：默认当“问描述”（注意：直接赋值）

        data['question_types'] = question_types                                  # data['question_types']=list[str] 固定字段：问句类型列表
        return data                                                              # 返回 data:dict -> {"args":..., "question_types":...}

    # ======================= 构造“实体词 -> 类型列表”映射：region_words -> wdtype_dict =======================
    def build_wdtype_dict(self):
        word_dict = dict()                                                       # word_dict:dict[str,list[str]]
        for word in self.region_words:                                           # word:str 遍历所有实体词
            word_dict[word] = []                                                 # 初始化类型列表

            if word in self.disease_words:
                word_dict[word].append('disease')                                # 词 -> disease

            if word in self.drug_words:
                word_dict[word].append('drug')                                   # 词 -> drug

            if word in self.food_words:
                word_dict[word].append('food')                                   # 词 -> food

            if word in self.symptom_words:
                word_dict[word].append('symptom')                                # 词 -> symptom

            if word in self.department_words:
                word_dict[word].append('department')                             # 词 -> department

            if word in self.check_words:
                word_dict[word].append('check')                                  # 词 -> check

            if word in self.producer_words:
                word_dict[word].append('producer')                               # 词 -> producer

        return word_dict                                                         # 返回 dict[str, list[str]]

    # ======================= 构造 AC 自动机：wordlist(list[str]) -> Automaton =======================
    def build_actree(self, wordlist):
        actree = ahocorasick.Automaton()                                         # actree:Automaton
        for index, word in enumerate(wordlist):                                  # index:int, word:str
            actree.add_word(word, (index, word))                                 # 注册关键词word；payload=(index, word)
        actree.make_automaton()                                                  # 构建自动机
        return actree                                                            # 返回 Automaton

    # ======================= 实体识别：question(str) -> final_dict(dict[str,list[str]]) =======================
    def check_medical(self, question):
        region_words = []                                                        # region_words:list[str] 命中的实体词（可能含子词/重复）

        for i in self.region_tree.iter(question):                                # i:tuple (end_index, (idx, word))
            word = i[1][1]                                                       # word:str 取出匹配到的实体词
            region_words.append(word)                                            # region_words += [word]

        stop_words = []                                                          # stop_words:list[str] 子词名单（用于剔除）
        for word1 in region_words:                                               # word1:str
            for word2 in region_words:                                           # word2:str
                if word1 in word2 and word1 != word2:                            # word1 是 word2 的子串（更短）
                    stop_words.append(word1)                                     # 加入待剔除列表

        final_words = [w for w in region_words if w not in stop_words]           # final_words:list[str] 剔除子词后的实体词
        final_dict  = {w: self.wdtype_dict.get(w) for w in final_words}          # final_dict:dict[str,list[str]] 词 -> 类型列表
        return final_dict                                                        # 返回 medical_dict 给 classify 用

    # ======================= 触发词检测：words(list[str]) + sent(str) -> bool =======================
    def check_word_in_sentence(self, words, sent):
        for word in words:                                                       # word:str
            if word in sent:                                                     # 子串命中
                return True                                                      # bool
        return False                                                             # bool

if __name__ == '__main__':
    qc = QuestionClassifier()                                                    # 初始化：构建词表、AC自动机、wdtype_dict
    while True:
        question = input('input an question:')                                   # question:str 用户输入
        data = qc.classify(question)                                             # data:dict 输出 {"args":..., "question_types":...}
        print(data)                                                              # 打印 data