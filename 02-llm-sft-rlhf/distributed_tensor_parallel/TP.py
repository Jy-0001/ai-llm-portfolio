# TP(Tensor Parellel)
    # 切分对象：模型参数 W（不切数据）
    # 每卡：只存 W 的一部分，看完整的同一份数据
    # 两种切分方式
        # 按行切分 W
            # 数据分布（forward开始时）：
                # GPU0只有X1，维度(b,s,h/N)  ← X被切开，每卡只有一部分
                # GPU1只有X2，维度(b,s,h/N)
                # W也被切开：GPU0有W1(h/N,h')，GPU1有W2(h/N,h')
            # Forward：
                # f（左紫条）：把X按列切成X1,X2分发给两卡
                # GPU0独立算：X1*W1 = Y1，维度(b,s,h')
                # GPU1独立算：X2*W2 = Y2，维度(b,s,h')
                # g（右紫条）：Y1+Y2 做AllReduce
                    # → 两卡都得到完整Y=Y1+Y2，维度(b,s,h')
                    # → ① 1次AllReduce
            # Backward：
                # 从后面层传回来∂L/∂Y（完整，两卡都有，不需要通讯）
                # 第一步：各卡算自己的∂L/∂Wi（更新参数用）
                    # GPU0：∂L/∂W1 = X1^T * ∂L/∂Y
                    # GPU1：∂L/∂W2 = X2^T * ∂L/∂Y
                    # 不需要通讯：GPU0自己有X1和∂L/∂Y，GPU1同理
                # 第二步：各卡算∂L/∂Xi（传给前一层用）
                    # GPU0：∂L/∂X1 = ∂L/∂Y * W1^T，维度(b,s,h/N)
                    # GPU1：∂L/∂X2 = ∂L/∂Y * W2^T，维度(b,s,h/N)
                    # f（左紫条）backward：直接concat拼接
                        # ∂L/∂X = concat[∂L/∂X1, ∂L/∂X2]，维度(b,s,h)
                        # 不需要AllReduce：两卡各算一部分，维度上拼接即可
                        # → ② All-Gather（拼接语义）
            # 通讯汇总：
                # forward：1次AllReduce（g处，Y1+Y2加和）
                # backward：1次All-Gather（f处，concat梯度）
        # 按列切分 W
            # 数据分布（forward开始时）：
                # GPU0有完整X，维度(b,s,h)  ← X被复制，每卡都有全量
                # GPU1有完整X，维度(b,s,h)
                # W被列切：GPU0有W1(h,h'/N)，GPU1有W2(h,h'/N)
            # Forward：
                # f（左紫条）：把完整X复制给两卡（identity，无通讯）
                # GPU0独立算：X*W1 = Y1，维度(b,s,h'/N)
                # GPU1独立算：X*W2 = Y2，维度(b,s,h'/N)
                # g（右紫条）：concat[Y1,Y2]做All-Gather
                    # → 两卡都得到完整Y，维度(b,s,h')
                    # → ① 1次All-Gather（注意：不是AllReduce，是拼接）
            # Backward：
                # 从后面层传回来∂L/∂Y（完整，两卡都有，不需要通讯）
                # g（右紫条）backward：直接把∂L/∂Y按列拆分发给两卡
                    # GPU0得到∂L/∂Y1，GPU1得到∂L/∂Y2
                    # 不需要通讯：split操作
                # 第一步：各卡算自己的∂L/∂Wi（更新参数用）
                    # GPU0：∂L/∂W1 = X^T * ∂L/∂Y1
                    # GPU1：∂L/∂W2 = X^T * ∂L/∂Y2
                    # 不需要通讯：各卡有完整X和自己那份∂L/∂Yi
                # 第二步：各卡算∂L/∂X（传给前一层用）
                    # GPU0：∂L/∂X|1 = ∂L/∂Y1 * W1^T，维度(b,s,h)
                    # GPU1：∂L/∂X|2 = ∂L/∂Y2 * W2^T，维度(b,s,h)
                    # 问题：X是完整的(b,s,h)，两卡对X都有梯度贡献，必须加和！
                    # f（左紫条）backward：做AllReduce
                        # ∂L/∂X = ∂L/∂X|1 + ∂L/∂X|2，维度(b,s,h)
                        # → ② 1次AllReduce
                    # 为什么是AllReduce不是All-Gather？
                        # 因为两卡算的∂L/∂X|1和∂L/∂X|2维度相同(b,s,h)
                        # 都是对完整X的梯度估计，需要加和合并
            # 通讯汇总：
                # forward：1次All-Gather（g处，concat Y1,Y2）
                # backward：1次AllReduce（f处，∂L/∂X加和）
        # 两种切分对比
            # 按行切分：forward AllReduce + backward All-Gather
            # 按列切分：forward All-Gather + backward AllReduce
            # 两种方式：各发生2次卡间通讯，总通讯量相同
            # 选哪种？看后面接的激活函数
                # 接非线性激活（如GELU）→ 用列切分（可以先激活再通讯）
                # 接线性或无激活 → 行切分也可以

    # 分层切分：
        # MLP层
            # 结构：Y = GELU(X * A) * B
                # A：(h, h')   第一个线性层
                # B：(h', h)   第二个线性层
                # GELU：非线性激活函数，夹在两层之间

            # 切分策略：A列切 + B行切
                # 为什么A用列切？
                    # GELU是非线性的：GELU(Y1+Y2) ≠ GELU(Y1) + GELU(Y2)
                    # 如果A行切 → forward时Y=Y1+Y2需要先AllReduce → 再过GELU → 多一次通讯
                    # 如果A列切 → 各卡独立得到Y1,Y2 → 各卡独立过GELU → 无需通讯
                # 为什么B用行切？
                    # A列切后，GELU(Y1)和GELU(Y2)分别在两卡上，维度(b,s,h'/N)
                    # B行切后，B1(h'/N,h)和B2(h'/N,h)与之维度配套，可以继续独立计算
                    # → A确定列切，B只能行切（维度决定的）

            # Forward：
                # f：复制X给两卡（列切分的f：identity，无通讯）
                # GPU0：X*A1 → GELU → Y1，维度(b,s,h'/N)
                # GPU1：X*A2 → GELU → Y2，维度(b,s,h'/N)
                # GPU0：Y1*B1 = Z1，维度(b,s,h)
                # GPU1：Y2*B2 = Z2，维度(b,s,h)
                # g：Z1+Z2 做AllReduce → 两卡都得到完整Z，维度(b,s,h)
                    # → ① 1次AllReduce

            # Backward：
                # 从后层传回∂L/∂Z（完整，两卡都有）
                # g backward：∂L/∂Z直接复制给两卡（无通讯）
                # 各卡独立算∂L/∂Bi（更新B用）：∂L/∂Bi = Yi^T * ∂L/∂Z
                # 各卡独立算∂L/∂Yi（传给GELU backward用）：∂L/∂Yi = ∂L/∂Z * Bi^T
                # 各卡独立过GELU backward → 得到∂L/∂(XAi)
                # 各卡独立算∂L/∂Ai（更新A用）：∂L/∂Ai = X^T * ∂L/∂(XAi)
                # 各卡算∂L/∂X（传给前一层用）：
                    # GPU0：∂L/∂X|1 = ∂L/∂(XA1) * A1^T，维度(b,s,h)
                    # GPU1：∂L/∂X|2 = ∂L/∂(XA2) * A2^T，维度(b,s,h)
                    # X被复制过 → 两卡梯度需加和
                    # f backward：AllReduce → ∂L/∂X = ∂L/∂X|1 + ∂L/∂X|2
                        # → ② 1次AllReduce

            # 通讯汇总：
                # forward：1次AllReduce（g处）
                # backward：1次AllReduce（f处）
                # 每个MLP层共2次AllReduce
        # Self-Attention层
            # 结构：MultiHead(Q,K,V) → concat → 线性层B
                # Q = X*Wq，K = X*Wk，V = X*Wv
                # 每个head独立做attention计算，最后concat
                # 天然适配TP：head本来就是独立计算的！

            # 切分策略：Wq/Wk/Wv列切 + B行切
                # Wq/Wk/Wv列切 → 每块GPU负责几个head
                    # 每head维度：(d_model, k_dim/N)
                    # 各卡独立做attention，无需通讯
                # B行切 → 与MLP的B行切完全一样的逻辑

            # Forward：
                # f：复制X给两卡（无通讯）
                # GPU0：用Wq1,Wk1,Wv1 算自己负责的head → 得到Y1，维度(b,s,h/N)
                # GPU1：用Wq2,Wk2,Wv2 算自己负责的head → 得到Y2，维度(b,s,h/N)
                # GPU0：Y1*B1 = Z1，维度(b,s,h)
                # GPU1：Y2*B2 = Z2，维度(b,s,h)
                # g：Z1+Z2 做AllReduce → 完整Z
                    # → ① 1次AllReduce

            # Backward：与MLP完全对称，不再赘述
                # → ② 1次AllReduce（f处，∂L/∂X加和）

            # 通讯汇总：每个Attention层共2次AllReduce

            # 实践约束：head总数必须能被GPU数整除
                # 否则每卡负责的head数不均等，浪费计算资源
        # Embedding层
            # Embedding分两部分：
                    # word embedding：维度(v, h)，v=词表大小（很大，需切分）
                    # positional embedding：维度(max_s, h)（不大，每卡完整拷贝）

                # 输入层word embedding：按行切分
                    # 原理：过embedding = 用token序号查表（取对应行的词向量）
                    # 切分方式举例（词表300，2卡）：
                        # GPU0维护词表[0, 150)
                        # GPU1维护词表[150, 299)
                    # 查表逻辑：
                        # token序号在自己范围内 → 正常返回词向量
                        # token序号不在范围内 → 返回全0向量
                    # 查完后AllReduce → 全0+正确词向量 = 正确结果
                        # GPU0查[0,212,7,9]结果：[ok, 0, ok, ok]
                        # GPU1查[0,212,7,9]结果：[0, ok, 0, 0]
                        # AllReduce加和 → [ok, ok, ok, ok] ✓

                # 输出层word embedding：按列切分
                    # 输入X(b,s,h) 乘以 WE(h,v)
                    # WE列切：GPU0有WE1(h,v/N)，GPU1有WE2(h,v/N)
                    # 各卡得到Y1(b,s,v/N)，Y2(b,s,v/N)
                    # 不做AllReduce，保持切分状态 → 直接传给Cross-Entropy层处理

                # ⚠️ 关键：输入层和输出层共享同一套word embedding
                    # backward时两层都会对word embedding算梯度
                    # 权重更新必须用两次梯度的总和
                    # PP深度=1（输入输出同一GPU）：自动累加，不用担心
                    # PP深度>1（输入输出在不同GPU）：权重更新前必须AllReduce word embedding梯度
        # Cross-Entropy层
            # 背景：输出层embedding后，Y被切分在各卡上，维度(b,s,v/N)
                # 目标：算softmax + cross entropy得到Loss

                # 朴素做法（通讯量大）：
                    # All-Gather Y1,Y2 → 拼成完整Y(b,s,v)
                    # 按行softmax → 与真值算cross entropy
                    # 通讯量：b*s*v （词表v很大时爆炸）

                # 优化做法：
                    # 第1步：各卡对自己的Yi按行求和 → GPU_sum(e)
                        # GPU0算e1=sum(exp(Y1))，维度(b,s)
                        # GPU1算e2=sum(exp(Y2))，维度(b,s)
                    # 第2步：AllReduce → e = e1+e2（softmax分母）
                        # 通讯量：b*s （远小于b*s*v）
                    # 第3步：各卡独立算 exp(Yi)/e → 得到各自负责部分的softmax概率
                    # 第4步：各卡与真值算cross entropy → 得到各自的loss（Li）
                    # 第5步：AllReduce → L = L1+L2（总Loss）
                        # 通讯量：N（GPU数，极小）
                    # 总通讯量：b*s + N（vs 朴素的b*s*v，大幅降低）

    # 经典部署：TP + DP 混合并行
        # 典型架构：
            # 同一台机器内（intra-node）：做TP
            # 不同机器间（inter-node）：做DP
            # DP中引入ZeRO做显存优化

        # 为什么TP放机器内，DP放机器间？核心：带宽需求不同
            # TP的backward特性：
                # 每层backward时所有TP卡必须做AllReduce
                # 下一层的backward依赖这次AllReduce的结果 → 强同步
                # → 必须等通讯完才能继续 → 要求极高带宽
                # → 放机器内（NVLink带宽高）
            # DP的backward特性：
                # 本层算完梯度就发出去做AllReduce
                # 同时继续往下一层做backward（不等结果）
                # → 通讯和计算可以重叠（异步）→ 带宽要求低
                # → 放机器间（跨机网络带宽低但够用）
