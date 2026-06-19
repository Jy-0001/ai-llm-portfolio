from question_classifier import *
from question_parser import *
from answer_search import *
from generator_module.chat_gpt import *
import time

# 红蜘蛛机器人综合问答类
class Red_Spider:
    def __init__(self):
        # 1:问题分类器
        print('初始化QuestionClassifier......')
        self.classifier = QuestionClassifier()
        # 2:问题解析器
        print('初始化QuestionParser......')
        self.parser = QuestionParser()
        # 3:答案搜索器
        print('初始化AnswerSearcher......')
        self.searcher = AnswerSearcher()
        
        # 4: ⽣成回复模块
        print('初始化ChatGPT......')
        self.generator = ChatGPT(flag, model_path)
        # 开幕词设置
        self.answer = 'hello, I am red spider ai assistant, please input your queries'
        print(self.answer)

    def chat_main(self, sentence):

        # 1:⾸先进⾏问题分类, 如果⽆法分类到症状, ⻝品, 药品相关问题上, 则直接返回客套话
        res_classify = self.classifier.classify(sentence)

        # 如果⽆法分类到症状, ⻝品, 药品等相关问题上, 则进⼊LLM⽣成(deepseek,gpt2,yuyuan,internlm,qwen-3b)
        if not res_classify:
            return self.generator.chat(sentence)

        # print('res_classify: ', res_classify)

        # 2: 对分类后的问题进⾏解析, 组装出neo4j查询语句
        res_sql = self.parser.parser_main(res_classify)
        if not res_sql:
            return self.generator.chat(sentence)

        # print('res_sql: ', res_sql)

        # 3: 利⽤查询语句, 直接调⽤答案搜索器查询neo4j, 得到最终答案
        final_answers = self.searcher.search_main(res_sql)

        # 无法查询到相关答案，则返回客套话；否则将若干答案分行返回
        if not final_answers:
            return self.generator.chat(sentence)
        else:
            return '\n'.join(final_answers)
        
if __name__ == '__main__':
    # 实例化红蜘蛛机器人
    print('初始化AI红蜘蛛......')
    start_time = time.time()
    
    #=============================================生成模型选择=============================================
    # flag = 'gpt2'
    # flag = 'yuyuan'
    # flag = 'intern'
    flag = 'qwen'
    # flag = 'deepseek'
    #=============================================生成模型选择=============================================


    model_path = './pretrain_model'
    red_spider = Red_Spider()
    end_time = time.time()
    print('初始化耗时{}s'.format(end_time - start_time))

    # 无限循环多轮对话
    while True:
        question = input('用户:')
        if question == 'Q' or question == 'q':
            break
        
        answer = red_spider.chat_main(question)
        print('\n')
        print('红蜘蛛：', answer)
        
