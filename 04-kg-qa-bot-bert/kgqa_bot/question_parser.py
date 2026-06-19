'''问题解析子任务'''
class QuestionParser:
    # 构建实体节点
    def build_entitydict(self, args):
        entity_dict = {}
        for arg, types in args.items():
            for type in types:
                if type not in entity_dict:
                    entity_dict[type] = [arg]
                else:
                    entity_dict[type].append(arg)

            return entity_dict
        
    # 解析主函数
    def parser_main(self, res_classify):
        args = res_classify['args']
        entity_dict = self.build_entitydict(args)
        question_types = res_classify['question_types']
        sqls = []
        for question_type in question_types:
            sql_ = {}
            sql_['question_type'] = question_type
            sql = []

            # 按照不同分类结果，组装不同cypher查询语句
            if question_type == 'disease_symptom':
                sql = self.sql_transfer(question_type, entity_dict.get('disease'))
            elif question_type == 'disease_food':
                sql = self.sql_transfer(question_type, entity_dict.get('disease'))
            elif question_type == 'disease_drug':
                sql = self.sql_transfer(question_type, entity_dict.get('disease'))

            if sql:
                sql_['sql'] = sql
                sqls.append(sql_)

        return sqls

    # 针对不同问题，分开进行处理
    def sql_transfer(self, question_type, entities):
        if not entities:
            return []
        
        # 查询语句
        sql = []
        # 查询疾病有哪些症状
        if question_type == 'disease_symptom':
            sql = ["match (m:Disease)-[r:has_symptom]->(n:Symptom) where m.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]
        # 查询疾病建议吃的东西
        elif question_type == 'disease_food':
            sql = ["match (m:Disease)-[r:recommand_eat]->(n:Food) where m.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]
        # 查询疾病常用药品
        elif question_type == 'disease_drug':
            sql = ["match (m:Disease)-[r:recommand_drug]->(n:Drug) where m.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]

        return sql
if __name__ == '__main__':
    qp = QuestionParser()
    print(qp)
