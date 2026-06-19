'''答案搜索子任务'''
import os 
import json
from neo4j import GraphDatabase
from config import NEO4J_CONFIG
import traceback

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
            try:
                final_answer = '{0}的症状包括:{1}'.format(subject, ';'.join(list(set(desc))[:self.num_limit]))
            except:
                final_answer = '对不起, 红蜘蛛AI⼩助理没有找到关于{0}的相关症状'.format(subject)
        elif question_type == 'symptom_disease':
            desc = [i['m.name']for i in answers]
            subject = answers[0]['n.name']
            try:
                final_answer = '症状{0}可能染上的疾病有:{1}'.format(subject, ';'.join(list(set(desc))[:self.num_limit]))
            except:
                final_answer = '对不起, 红蜘蛛AI⼩助理没有找到关于{0}的相关症状'.format(subject)
        elif question_type == 'disease_cause':
            desc = [i['m.name']for i in answers]
            subject = answers[0]['m.name']
            try:
                final_answer = '疾病{0}可能的成因有:{1}'.format(subject, ';'.join(list(set(desc))[:self.num_limit]))
            except:
                final_answer = '对不起, 红蜘蛛AI⼩助理没有找到关于{0}的相关症状'.format(subject)
        elif question_type == 'disease_prevent':
            desc = [i['m.prevent'] for i in answers]
            subject = answers[0]['m.name']
            try:
                final_answer = '{0}的预防措施包括:{1}'.format(subject, ';'.join(list(set(desc))[:self.num_limit]))
            except:
                final_answer = '对不起, 红蜘蛛AI⼩助理没有找到关于{0}的预防措施'.format(subject)
        elif question_type == 'disease_lasttime':
            desc = [i['m.cure_lasttime'] for i in answers]
            subject = answers[0]['m.name']
            try:
                final_answer = '{0}治疗可能持续的周期为:{1}'.format(subject, ';'.join(list(set(desc))[:self.num_limit]))
            except:
                final_answer = '对不起, 红蜘蛛AI⼩助理没有找到关于{0}治疗的持续周期'.format(subject)
        elif question_type == 'disease_cureway':
            desc = [';'.join(i['m.cure_way']) for i in answers]
            subject = answers[0]['m.name']
            try:
                final_answer = '{0}可以尝试如下治疗:{1}'.format(subject, ';'.join(list(set(desc))[:self.num_limit]))
            except:
                final_answer = '对不起, 红蜘蛛AI⼩助理没有找到关于{0}的治疗⽅法'.format(subject)
        elif question_type == 'disease_cureprob':
            desc = [i['m.cured_prob'] for i in answers]
            subject = answers[0]['m.name']
            try:
                final_answer = '{0}治愈的概率为(仅供参考):{1}'.format(subject, ';'.join(list(set(desc))[:self.num_limit]))
            except:
                final_answer = '对不起, 红蜘蛛AI⼩助理没有找到关于{0}的治愈概率'.format(subject)
        elif question_type == 'disease_easyget':
            desc = [i['m.easy_get'] for i in answers]
            subject = answers[0]['m.name']
            try:
                final_answer = '{0}的易感⼈群包括:{1}'.format(subject, ';'.join(list(set(desc))[:self.num_limit]))
            except:
                final_answer = '对不起, 红蜘蛛AI⼩助理没有找到关于{0}的易感⼈群'.format(subject)
        elif question_type == 'disease_desc':
            desc = [i['m.desc'] for i in answers]
            subject = answers[0]['m.name']
            try:
                final_answer = '{0}, 了解⼀下:{1}'.format(subject, ';'.join(list(set(desc))[:self.num_limit]))
            except:
                final_answer = '对不起, 红蜘蛛AI⼩助理没有找到关于{0}的疾病基本描述'.format(subject)
        elif question_type == 'disease_accompany':
            desc1 = [i['n.name'] for i in answers]
            desc2 = [i['m.name'] for i in answers]
            subject = answers[0]['m.name']
            desc = [i for i in desc1 + desc2 if i != subject]
            try:
                final_answer = '{0}的并发症包括:{1}'.format(subject, ';'.join(list(set(desc[0]))[:self.num_limit]))
            except:
                final_answer = '对不起, 红蜘蛛AI⼩助理没有找到关于{0}的相关并发症'.format(subject)

        elif question_type == 'disease_belong':
            desc = [i['m.cure_department'] for i in answers]
            subject = answers[0]['m.name']
            try:
                final_answer = '疾病{0}的科室为：{1}'.format(subject, ';'.join(list(set(desc[0]))[:self.num_limit]))
            except:
                final_answer = '对不起, 红蜘蛛AI⼩助理没有找到关于{0}的科室信息'.format(subject)

        elif question_type == 'disease_not_food':
            desc = [i['n.name'] for i in answers]
            subject = answers[0]['m.name']
            try:
                final_answer = '{0}忌⻝的⻝物包括有:{1}'.format(subject, ';'.join(list(set(desc))[:self.num_limit]))
            except:
                final_answer = '对不起, 红蜘蛛AI⼩助理没有找到关于{0}的忌吃⻝物'.format(subject)
        elif question_type == 'disease_do_food':
            do_desc = [i['n.name'] for i in answers if i['r.name'] == '宜吃']
            subject = answers[0]['m.name']
            try:
                final_answer = '{0}宜⻝的⻝物包括有:{1}'.format(subject, ';'.join(list(set(do_desc))[:self.num_limit]))
            except:
                final_answer = '对不起, 红蜘蛛AI⼩助理没有找到关于{0}的宜吃⻝物'.format(subject)
        elif question_type == 'food_not_disease':
            desc = [i['m.name'] for i in answers]
            subject = answers[0]['n.name']
            try:
                final_answer = '患有{0}的⼈最好不要吃:{1}'.format(';'.join(list(set(desc))[:self.num_limit]), subject)
            except:
                final_answer = '对不起, 红蜘蛛AI⼩助理没有找到忌吃{0}的相关疾病'.format(subject)
        elif question_type == 'food_do_disease':
            desc = [i['m.name'] for i in answers]
            subject = answers[0]['n.name']
            try:
                final_answer = '患有{0}的⼈建议多试试:{1}'.format(';'.join(list(set(desc))[:self.num_limit]), subject)
            except:
                final_answer = '对不起, 红蜘蛛AI⼩助理没有找到宜吃{0}的相关疾病'.format(subject)
        elif question_type == 'disease_food':
            desc = [i['n.name'] for i in answers]
            subject = answers[0]['m.name']
            try:
                final_answer = '{0}推荐⻝谱包括:{1}'.format(subject, ';'.join(list(set(desc))[:self.num_limit]))
            except:
                final_answer = '对不起, 红蜘蛛AI⼩助理没有找到{0}的推荐⻝谱'.format(subject)
        elif question_type == 'disease_drug':
            desc = [i['n.name'] for i in answers]
            subject = answers[0]['m.name']
            try:
                final_answer = '{0}推荐的药品包括:{1}'.format(subject, ';'.join(list(set(desc))[:self.num_limit]))
            except:
                final_answer = '对不起, 红蜘蛛AI⼩助理没有找到{0}的推荐药品'.format(subject)
        elif question_type == 'drug_disease':
            desc = [i['m.name'] for i in answers]
            subject = answers[0]['n.name']
            try:
                final_answer = '{0}主治的疾病有:{1}, 可以试试'.format(subject, ';'.join(list(set(desc))[:self.num_limit]))
            except:
                final_answer = '对不起, 红蜘蛛AI⼩助理没有找到药品{0}的主治疾病'.format(subject)
        elif question_type == 'disease_check':
            desc = [i['n.name'] for i in answers]
            subject = answers[0]['m.name']
            try:
                final_answer = '{0}通常可以通过以下⽅式检查出来:{1}'.format(subject, ';'.join(list(set(desc))[:self.num_limit]))
            except:
                final_answer = '对不起, 红蜘蛛AI⼩助理没有找到{0}的有效检出的⽅法'.format(subject)
        elif question_type == 'check_disease':
            desc = [i['m.name'] for i in answers]
            subject = answers[0]['n.name']
            try:
                final_answer = '通常可以通过{0}检查出来的疾病有:{1}'.format(subject, ';'.join(list(set(desc))[:self.num_limit]))
            except:
                final_answer = '对不起, 红蜘蛛AI⼩助理没有找到{0}可以有效检出的疾病'.format(subject)
        return final_answer


if __name__ == '__main__':
    ans = AnswerSearcher()
    print(ans)










