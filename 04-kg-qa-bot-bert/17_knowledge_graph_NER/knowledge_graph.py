# 知识图谱：
    # 基本概念：
        # 重要指标：
            # QPS: Query Per Second   (每秒钟能处理的请求数 --- 工业界的一线)
            # RT: Request Time  (每条请求的耗时)
        # 三元组：SPO
            # 主体: Subject
            # 谓词: Predicate
            # 客体: Object
        # 信息抽取任务：
            # NER实体识别：Named Entity Recognition：识别Subject和Object
                # 方法：
                    # 基线模型BiLSTM + CRF
                    # IDCNN, BERT, BERT + CRF：见IDCNN.py
                    # FLAT
                # 标注体系：
                    # BIO:
                        # B:begin，实体开始
                        # I:inside，实体中间部分
                        # O:outside，非实体
                    # BIEO
                        # E:end，实体结束
                    # BIEOS
                        # S:single，单个字符的实体
            # RE关系抽取：识别Subject, Predicate和Object
            # EE事件抽取：识别SOP以及时间，地点，时态
            # 图数据库的构建
            # 更精细化的操作
        # 序列标注任务：
            #CRF条件随机场：Conditional Random Field


