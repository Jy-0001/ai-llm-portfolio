'''答案搜索子任务'''
import os 
import json
from neo4j import GraphDatabase
from config import NEO4J_CONFIG

# 答案搜索的主类
class AnswerSearcher:
    def __init__(self):
        self.num_limit = 10
        self.driver = GraphDatabase.driver( **NEO4J_CONFIG)

    # 执行cypher查询，并返回相应结果
    def search_main(self, sqls):
        final_answers = []

        # 开启会话
        with self.driver.session() as session:
            for sql_ in sqls:
                question_type = sql_['question_type']
                queries = sql_['sql']
                answers = []

                # 遍历所有的查询cypher，依次执行，并将结果逐个添加进列表中
                for query in queries:
                    ress = session.run(query).data()
                    answers += ress

                # 调用精准回复模板
                final_answer = self.answer_prettify(question_type, answers)
                if final_answer:
                    final_answers.append(final_answer)

        return final_answers

    # 根据对应的question_type, 调用相应的回复模板
    def answer_prettify(self, question_type, answers):
        fianl_answer = []
        if not answers:
            return ''
        if question_type == 'disease_symptom':
            desc = [i['n.name']for i in answers]
            subject = answers[0]['m.name']
            final_answer = '{0}的症状包括:{1}'.format(subject, ';'.join(list(set(desc))[:self.num_limit]))

        elif question_type == 'disease_food':
            desc = [i['n.name']for i in answers]
            subject = answers[0]['m.name']
            final_answer = '{0}推荐食谱包括:{1}'.format(subject, ';'.join(list(set(desc))[:self.num_limit]))
        
        elif question_type == 'disease_drug':
            desc = [i['n.name']for i in answers]
            subject = answers[0]['m.name']
            final_answer = '{0}推荐的药品包括:{1}'.format(subject, ';'.join(list(set(desc))[:self.num_limit]))

        return final_answer

if __name__ == '__main__':
    ans = AnswerSearcher()
    print(ans)










