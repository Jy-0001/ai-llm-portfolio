# 红蜘蛛v1搭建
    # 代码实现：
        # 问题分类子任务:将问题分类到红蜘蛛可以⽀持回答的若⼲⼦类中, 有利于后续查询和回答模板的编写.
            # 见question_classifier.py
        # 问题解析子任务:对症状, ⻝品, 药品进⾏neo4j查询的cypher语句组装和解析.
            # 见question_parser.py
        # 答案搜索子任务:完成具体的查询, 并组装回复模板
            # 见answer_search.py
        # 核心模块：见chatbot.py