'''问题解析子任务'''
class QuestionParser:
    # ======================= 把 “args(实体词->类型列表)” 变成 “entity_dict(类型->实体词列表)” =======================
    def build_entitydict(self, args):
        entity_dict = {}                                                        # entity_dict:dict[str, list[str]] 最终结构：类型 -> 该类型下的实体词列表
        for arg, types in args.items():                                          # arg:str 实体词；types:list[str] 该实体词的类型列表
            for type in types:                                                   # type:str 单个类型（disease/drug/food/symptom/check/…）
                if type not in entity_dict:                                      # 如果该类型第一次出现
                    entity_dict[type] = [arg]                                    # 初始化：entity_dict[type]=list[str]，先放一个实体词
                else:
                    entity_dict[type].append(arg)                                # 已存在：追加实体词到 list[str]

        return entity_dict                                                   # ⚠️注意：这里缩进导致“只处理第一个 arg 就 return”（逻辑bug，数据会不完整）

    # ======================= 主解析：res_classify(dict) -> sqls(list[dict]) =======================
    def parser_main(self, res_classify):
        args = res_classify['args']                                              # args:dict[str,list[str]] 例：{"高血压":["disease"],"失眠":["symptom"]}
        entity_dict = self.build_entitydict(args)                                # entity_dict:dict[str,list[str]] 例：{"disease":["高血压"],"symptom":["失眠"]}（但受上面bug影响可能缺）
        question_types = res_classify['question_types']                          # question_types:list[str] 例：["disease_symptom","disease_food"]
        sqls = []                                                                # sqls:list[dict] 每个元素包含 question_type + sql(list[str])

        for question_type in question_types:                                     # question_type:str 遍历每一种问句类型
            sql_ = {}                                                            # sql_:dict 当前问句类型对应的“查询包”
            sql_['question_type'] = question_type                                # 记录类型：str
            sql = []                                                             # sql:list[str] 存放该类型对应的 cypher 语句（可能多条）

            # ===== 选择“用哪一类实体”来拼 cypher：核心是 entity_dict.get('xxx') -> entities(list[str]) =====
            if question_type == 'disease_symptom':
                sql = self.sql_transfer(question_type, entity_dict.get('disease'))   # entities:list[str] 疾病名列表
            elif question_type == 'symptom_disease':
                sql = self.sql_transfer(question_type, entity_dict.get('symptom'))   # entities:list[str] 症状名列表
            elif question_type == 'disease_cause':
                sql = self.sql_transfer(question_type, entity_dict.get('disease'))   # diseases
            elif question_type == 'disease_accompany':
                sql = self.sql_transfer(question_type, entity_dict.get('disease'))   # diseases
            elif question_type == 'disease_not_food':
                sql = self.sql_transfer(question_type, entity_dict.get('disease'))   # diseases
            elif question_type == 'disease_do_food':
                sql = self.sql_transfer(question_type, entity_dict.get('disease'))   # diseases
            elif question_type == 'food_not_disease':
                sql = self.sql_transfer(question_type, entity_dict.get('food'))      # foods
            elif question_type == 'food_do_disease':
                sql = self.sql_transfer(question_type, entity_dict.get('food'))      # foods
            elif question_type == 'disease_food':
                sql = self.sql_transfer(question_type, entity_dict.get('disease'))   # diseases
            elif question_type == 'disease_drug':
                sql = self.sql_transfer(question_type, entity_dict.get('disease'))   # diseases
            elif question_type == 'drug_disease':
                sql = self.sql_transfer(question_type, entity_dict.get('drug'))      # drugs
            elif question_type == 'disease_check':
                sql = self.sql_transfer(question_type, entity_dict.get('disease'))   # diseases
            elif question_type == 'check_disease':
                sql = self.sql_transfer(question_type, entity_dict.get('check'))     # checks
            elif question_type == 'disease_prevent':
                sql = self.sql_transfer(question_type, entity_dict.get('disease'))   # diseases
            elif question_type == 'disease_lasttime':
                sql = self.sql_transfer(question_type, entity_dict.get('disease'))   # diseases
            elif question_type == 'disease_cureway':
                sql = self.sql_transfer(question_type, entity_dict.get('disease'))   # diseases
            elif question_type == 'disease_cureprob':
                sql = self.sql_transfer(question_type, entity_dict.get('disease'))   # diseases
            elif question_type == 'disease_easyget':
                sql = self.sql_transfer(question_type, entity_dict.get('disease'))   # diseases
            elif question_type == 'disease_desc':
                sql = self.sql_transfer(question_type, entity_dict.get('disease'))   # diseases

            elif question_type == 'disease_belong':
                sql = self.sql_transfer(question_type, entity_dict.get('disease'))   # diseases

            if sql:                                                               # 如果该类型确实拼出了查询语句（sql 非空）
                sql_['sql'] = sql                                                 # sql_:dict 加入字段 'sql'，值为 list[str]
                sqls.append(sql_)                                                 # sqls:list[dict] 追加一个“查询包”

        return sqls                                                               # 返回：list[{"question_type":..., "sql":[...]}]

    # ======================= 把 (question_type, entities) -> cypher语句list[str] =======================
    def sql_transfer(self, question_type, entities):
        if not entities:                                                          # entities 为空或 None
            return []                                                             # 返回空 list[str]（表示无法拼 SQL）

        sql = []                                                                  # sql:list[str] 存 cypher 语句（按实体数量生成多条）

        # ===== disease_symptom：输入 entities=list[str] 疾病名 -> 输出 list[str] 每个疾病一条 cypher =====
        if question_type == 'disease_symptom':
            sql = ["MATCH (m:Disease)-[r:has_symptom]->(n:Symptom) where m.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]  # list[str]

        # ===== disease_food：疾病 -> 推荐吃 =====
        elif question_type == 'disease_food':
            sql = ["MATCH (m:Disease)-[r:recommand_eat]->(n:Food) where m.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]     # list[str]

        # ===== disease_drug：疾病 -> 推荐药 =====
        elif question_type == 'disease_drug':
            sql = ["MATCH (m:Disease)-[r:recommand_drug]->(n:Drug) where m.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]    # list[str]

        # ===== disease_prevent：疾病节点属性 prevent =====
        elif question_type == 'disease_prevent':
            sql = ["MATCH (m:Disease) where m.name = '{0}' return m.name, m.prevent".format(i) for i in entities]                                       # list[str]

        # ===== disease_cause：疾病节点属性 cause =====
        elif question_type == 'disease_cause':
            sql = ["MATCH (m:Disease) where m.name = '{0}' return m.name, m.cause".format(i) for i in entities]                                         # list[str]

        # ===== disease_lasttime：疾病节点属性 cure_lasttime =====
        elif question_type == 'disease_lasttime':
            sql = ["MATCH (m:Disease) where m.name = '{0}' return m.name, m.cure_lasttime".format(i) for i in entities]                                # list[str]

        # ===== disease_cureprob：疾病节点属性 cured_prob =====
        elif question_type == 'disease_cureprob':
            sql = ["MATCH (m:Disease) where m.name = '{0}' return m.name, m.cured_prob".format(i) for i in entities]                                   # list[str]

        # ===== disease_cureway：疾病节点属性 cure_way =====
        elif question_type == 'disease_cureway':
            sql = ["MATCH (m:Disease) where m.name = '{0}' return m.name, m.cure_way".format(i) for i in entities]                                     # list[str]

        # ===== disease_easyget：疾病节点属性 easy_get =====
        elif question_type == 'disease_easyget':
            sql = ["MATCH (m:Disease) where m.name = '{0}' return m.name, m.easy_get".format(i) for i in entities]                                     # list[str]

        # ===== disease_desc：疾病节点属性 desc =====
        elif question_type == 'disease_desc':
            sql = ["MATCH (m:Disease) where m.name = '{0}' return m.name, m.desc".format(i) for i in entities]                                         # list[str]

        # ===== disease_belong：疾病节点属性 department =====
        elif question_type == 'disease_belong':
            sql = ["MATCH (m:Disease) where m.name = '{0}' return m.name, m.cure_department".format(i) for i in entities]         # list[str]

        # ===== symptom_disease：症状 -> 可能的疾病（反向查）=====
        elif question_type == 'symptom_disease':
            sql = ["MATCH (m:Disease)-[r:has_symptom]->(n:Symptom) where n.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]    # list[str]

        # ===== disease_accompany：并发症（双向拼两条list再相加）=====
        elif question_type == 'disease_accompany':
            sql1 = ["MATCH (m:Disease)-[r:acompany_with]->(n:Disease) where m.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]  # list[str]
            sql2 = ["MATCH (m:Disease)-[r:acompany_with]->(n:Disease) where n.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]  # list[str]
            sql = sql1 + sql2                                                     # list[str] 合并：同一问题类型生成两组查询

        # ===== disease_not_food：疾病 -> 忌口 =====
        elif question_type == 'disease_not_food':
            sql = ["MATCH (m:Disease)-[r:no_eat]->(n:Food) where m.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]             # list[str]

        # ===== disease_do_food：疾病 -> 推荐吃 =====
        elif question_type == 'disease_do_food':
            sql = ["MATCH (m:Disease)-[r:do_eat]->(n:Food) where m.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]             # list[str]

        # ===== food_not_disease：已知食物(忌口项) -> 哪些病忌 =====
        elif question_type == 'food_not_disease':
            sql = ["MATCH (m:Disease)-[r:no_eat]->(n:Food) where n.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]             # list[str]

        # ===== food_do_disease：已知食物(推荐项) -> 哪些病推荐 =====
        elif question_type == 'food_do_disease':
            sql = ["MATCH (m:Disease)-[r:do_eat]->(n:Food) where n.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]             # list[str]

        # ===== disease_drug：⚠️重复分支（上面已经有 elif question_type == 'disease_drug'）=====
        elif question_type == 'disease_drug':
            sql1 = ["MATCH (m:Disease)-[r:common_drug]->(n:Drug) where m.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]       # list[str]
            sql2 = ["MATCH (m:Disease)-[r:recommand_drug]->(n:Drug) where m.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]    # list[str]
            sql = sql1 + sql2                                                     # list[str] 合并

        # ===== drug_disease：药 -> 治什么病（同理两组查询合并）=====
        elif question_type == 'drug_disease':
            sql1 = ["MATCH (m:Disease)-[r:common_drug]->(n:Drug) where n.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]       # list[str]
            sql2 = ["MATCH (m:Disease)-[r:recommand_drug]->(n:Drug) where n.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]    # list[str]
            sql = sql1 + sql2                                                     # list[str]

        # ===== disease_check：病 -> 需要做什么检查 =====
        elif question_type == 'disease_check':
            sql = ["MATCH (m:Disease)-[r:need_check]->(n:Check) where m.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]         # list[str]

        # ===== check_disease：检查 -> 提示哪些病 =====
        elif question_type == 'check_disease':
            sql = ["MATCH (m:Disease)-[r:need_check]->(n:Check) where n.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]         # list[str]

        
        return sql                                                                 # 返回 list[str]：每个实体一条或多条 cypher

if __name__ == '__main__':
    qp = QuestionParser()
    print(qp)
