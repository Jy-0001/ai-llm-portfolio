#知识蒸馏：
    # 目的：缩小模型，节省储存空间、算力等
    # 原理（核心思路：模仿教师模型的“输出分布/行为”而不是只学硬标签）：结构图见课件pdf
        # 1）先准备一个训练好的教师模型 Teacher（通常更大、更强）。
        # 2）训练一个学生模型 Student（更小），让它在同一输入上“尽量输出得像 Teacher”。
            # - Teacher 会对每个类别给出一组 logits/概率分布（不是只给一个正确类别）。
            # - 这个“软分布”包含类与类之间的相似性信息（暗知识 / dark knowledge），比硬标签信息更丰富。
        # 3）学生的损失通常由两部分组成（可加权）：
            # - 蒸馏损失：让 Student 的输出分布接近 Teacher（常用 KL 散度 / 交叉熵，配合温度 T）。
            # - 监督损失：让 Student 也对真实标签正确（标准 CrossEntropy）。
        # 4）训练完成后，只保留 Student 用于部署；Teacher 仅用于训练阶段提供“指导”。

        # 教师网络（大模型）：表达能力强，输出更“平滑/可靠”，能提供软标签指导。
        # 学生网络（小模型）：参数更少，但通过模仿 Teacher 的输出分布，往往能比“只学硬标签”训练得更好。
    # 公式：L=(1−α)CE(y,p)+αKL(q,p)
        #L：loss（损失函数，Loss）。训练时要最小化的目标数值。
        #α：alpha（混合系数，权重）。α∈[0,1] 时最常见。
            # (1−α)：给“硬标签损失”的权重
            # α：给“软标签蒸馏损失”的权重
            # α 越大 → 越强调“模仿老师”；越小 → 越强调“贴合真实标签”。
        # CE(⋅,⋅)：Cross Entropy（交叉熵，交叉熵损失）。它衡量“目标分布”和“预测分布”之间差异。
            #对输入有限制：要求label必须是one-hot格式的
        # KL(⋅,⋅)：KL 散度（Kullback-Leibler divergence）。它衡量“目标分布”和“预测分布”之间差异。
        # y：真实标签（ground-truth / hard label）的“目标分布”。通常是 one-hot：正确类位置为 1，其余为 0。
        # q：教师模型输出的“软标签分布”（soft target / teacher distribution）。不是 one-hot，而是每一类都有概率，比如“猫 0.6、狗 0.3、狐狸 0.1”。
            #q：需用有温度系数T参与的softmax计算得出：q_i = exp(logit_i / T) / sum(exp(logit_j / T))
                #T: 温度系数，通常取 1~10。
                    # 如果将T值取1, softmax-T公式就成为softmax公式, 根据logits输出各个类别的概率.
                    # 如果T越接近于0, 则最⼤值会越接近1, 其他值会接近0, 类似于退化成one-hot编码.
                    # 如果T越大, 则输出的结果分布越平缓, 相当于标签平滑的原理, 起到保留相似信息的作⽤.
                    # 如果T趋于⽆穷⼤, 则演变成均匀分布.

        # p：学生模型输出的预测分布（student prediction）。一般是学生 logits 过 softmax 得到的概率分布。
    # 方式：
        #模型压缩：将复杂教师网络的知识传递给简单学生网络，
        #同构蒸馏：教师模型与学生模型结构相同
        #集成蒸馏：模型之间进行集成，将多个教师模型知识传递给学生模型
        #大规模蒸馏：需要蒸馏出的Student模型往往需要⼤量的样本训练才能逼近⼤模型的结果
    # 两个模型类的代码：
        #Teacher模型代码：使用bert模型，见bert.py

        #Student模型代码：使用TextCnn模型，见textCNN.py
            #编写训练函数，测试函数，评估函数：见train_eval.py
            #编写运行主函数：见run.py
    #运行：
        #第一步：训练Teacher模型（此处为bert）
            # # 切换到主训练函数所在的路径下
            # cd /home/ec2-user/toutiao/bert_distil/src/
            # # 直接在命令⾏运⾏训练Teacher模型的代码
            # python run.py --task trainbert
        #第二步：训练Student模型（采用知识蒸馏的模式）
            #设定Config中的重要参数：见textCNN.py
                # # 模型迭代3轮
                # self.num_epochs = 3
                # # 卷积核尺⼨分别选2, 3, 4
                # self.filter_sizes = (2, 3, 4)
                # # 卷积核的个数512
                # self.num_filters = 512
            #调用：
                # # 切换到主训练函数所在的路径下
                # cd /home/ec2-user/toutiao/bert_distil/src/
                # # 直接在命令⾏运⾏训练Student模型的代码
                # python run.py --task train_kd
    


