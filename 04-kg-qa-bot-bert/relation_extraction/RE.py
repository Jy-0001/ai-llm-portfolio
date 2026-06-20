# multi-head-selection多头选择模型：
    # 学习目标：清晰代码层面实体间关系抽取的原理即可（重点关注sigmoid函数在这个过程中的作用）
    
    # 说明：可以将NER任务和RE任务合⼆为⼀, 作为⼀个整体来看待, 这也是多头的意义所在
        # 原理：基于尾部对齐（Tail-to-Tail）的多头选择机制。模型建立从“主体实体尾部（关系的发起者） Token（Subject Tail）”到“客体实体尾部（关系的接受者） Token（Object Tail）”的映射，并在对应的多维得分矩阵中编码关系类别（Relation Label）。
    # Architecture:
        # Embedding Layer
        # BiLSTM Layer
        # CRF Layer
        # Label Embeddings
        # Sigmoid Layer
        # Heads Relations
    # 代码实现：
        # 第⼀步: 查看数据集
            # 训练集数据train_data.json
            # 验证集数据dev_data.json
            # 词典数据word_vocab.json
            # 关系词典数据relation_vocab.json
            # 标注字典数据bio_vocab.json
        # 第⼆步: 构建数据迭代器:见models/selection_loader.py
        # 第三步: 构建模型类:见models/selection.py
        # 第四步: 编写评估指标函数:见metrics/F1_score.py
        # 第五步: 编写运⾏主函数:见main.py









