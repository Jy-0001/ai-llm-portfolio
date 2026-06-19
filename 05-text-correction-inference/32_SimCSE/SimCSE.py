# SimCSE模型Simple Contrastive Learning of Sentence Embeddings
    # 说明：属于对比学习，因为简化了经典对比学习框架，所以叫simple，大量的应用在cv，nlp中
        # 对比学习Contrastive Learning：不直接让模型预测“类别”，而是让它学会“谁跟谁更像、谁跟谁更不像”
    # 架构图
        # 无监督SimCSE
            # 核⼼在于对同⼀个样本执⾏2次forward pass, ⽽样本增强的操作通过dropout mask来实现
        # 有监督SimCSE
            # 核⼼在于对于同⼀个样本, 有不同的label⽂本：
                # 当label=entailment, 则对应的⽂本为和输⼊⽂本保持蕴含关系, 即为正样本.
                # 当label=contradiction, 则对应的⽂本为和输⼊⽂本保持⽭盾关系, 即为负样本.
    # 原理：
        # 前提：对比学习中，对于学到的embedding分布的两大核心指标：
            # Alignment：衡量同一个语义的两种view在dropout后的距离，最好要离得很近。
                # 关注点：关注的是相似样本间的距离, 度量⽅法是正例间的距离的期望
                # 公式：ℓalign​=E(x,x+)∼ppos​||f(x)−f(x+)||^2
            # Uniformity：衡量不同语义的句向量，在单位超球面上分布得是否均匀，最好比较均匀，不要挤成一坨。
                # 关注点：关注的是不相似样本间的距离, 度量⽅法时负例间的距离的指数期望
                # 公式：ℓuniform​=logEx,y∼pdata​​ e^(−2||f(x)−f(y)||^2)
        # infoNCE loss损失函数：info Noise Contrastive Estimation loss
            # 定义：对于⼀个拥有N个样本batch中的样本i:
            # ℓi​=−log{(e^(sim(hi​,hi+​)/τ))​ / (∑j=1,N​ e^(sim(hi​,hj+​)/τ))}
                # i:当前样本的编号
                # N：这一批里有多少个样本（batch size）
                # hi:第i个样本的表示向量（embedding）
                # hi+:第i个样本的正样本表示（同一句话的另一种view，比如另一种dropout；或监督SimCSE里的entailment句）
                # hj+:第j个样本的正样本表示；当j≠i时，它们对hi​来说就是负样本候选
                # sim(⋅,⋅)：相似度函数（常用余弦相似度或点积）
                # τ（tau）：温度系数，调节对困难样本 (Hard negative) 的关注程度，让softmax更“尖/钝”
                    # τ 小 → 更尖锐（更强拉开差距），但是不能太小！
                    # τ 大 → 更平滑
    # 核心：
        # 无监督数据增强：
            # 正例样本：让同⼀个样本 经历神经⽹络, 在dropout加持下进⾏两次forward计算, 得到2个输出zi,zi'
            # 负例样本：在同⼀个batch内部选取不同xj增强后得到的xj', 就得到了负例样本(zi,zj')
            # 公式：ℓi​=−log exp(sim(hizi​​,hizi+​​)/τ)​ / ∑j=1N​exp(sim(hizi​​,hjzj+​​)/τ)
                # hizi​：第i个句子经过一次 dropout 得到的向量
                # hizi+：同一句子再跑一次 dropout 得到的另一个向量
                # hjzj+：batch 里其他句子的向量（它们各自也来自一次 dropout view）
    # 代码实现：
        # 第1步: 查看数据集
        # 第2步: ⼯具类代码实现
        # 第3步: SimCSE模型类实现
        # 第4步: 数据类实现
        # 第5步: 损失函数实现
        # 第6步: 训练函数实现
        # 第7步: 推理类实现
        # 第8步: 主运⾏函数实现





