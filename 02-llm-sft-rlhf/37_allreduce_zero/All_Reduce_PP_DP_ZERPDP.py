# All-Reduce集群通信
    # 目的：让每个设备上的矩阵中的每一个位置的数值都是所有设备上对应位置的数值之和!
    # 原子操作拆解：基于ring环状通信算法
        #Reduce-scatter：
            # Reduce（汇总）+ Scatter（分发）
            # 每张卡只拿到自己负责那部分的汇总结果
            # 通讯量：Φ
        #All-gather：
            # All（所有）+ Gather（收集）
            # 每张卡把自己的部分广播给所有卡
            # 通讯量：Φ
    # ring环状通信算法：
        # 核心思想：充分利用设备间带宽，每张卡的出入口同时传输
        # 执行步骤：p-1 步完成 reduce-scatter
        # 通讯时间：(p-1)·V/(p·β)
            # V为数据总量
            # β为带宽
            # p足够大时近似 V/β
            # 结论：时间与设备数 p 无关
        # 通讯量：
            # 单设备通讯量：(p-1)·V/p
            # 总通讯量：(p-1)·V
        # 总通讯量：(p-1)·V，与设备数 p 成正比
        # 最优性：理论上不可能有更好的算法（大V场景）
        # 实现：NVIDIA NCCL 已内置
    # 关键洞察：通讯量 = 冗余显存
        # 每张卡的冗余显存 = 每张卡的通讯量
        # 增加并行宽度 p 是双刃剑
            # 好：每卡处理数据变小（V/p），计算更快
            # 坏：通讯量和冗余显存都按正比增长

# 3D并行计算
    # 方式：
        # DP：Data Parallel 数据并行 → 切数据
            # 整体流程
                # 1. 每块 GPU 拷贝一份完整模型参数 W
                # 2. 把一个 batch 数据均匀切分给各 GPU
                # 3. 每卡各自 FWD + BWD，算出各自的梯度
                # 4. AllReduce 聚合梯度
                # 5. 用聚合后的梯度更新参数
                # 6. 所有卡参数保持一致
            # 简单用法：torch.nn.DataParallel(model, device_ids=[0,1,2,3])
            # 推荐用法：torch.nn.parallel.DistributedDataParallel (DDP)
            # 存在的问题
                # 每张卡都存完整 W + G + O
                # 大模型下显存爆炸 → 引出 ZeRO
        # TP：Tensor Parallel 张量并行 → 切层内:见38
        # PP：Pipeline Parallel 流水线并行 → 切层间
            # 为什么需要：模型太大，单卡装不下
            # 核心做法：把不同层放到不同 GPU
                # 例：4层模型 → 分给 3 个 GPU
                    # Device 0：L1
                    # Device 1：L2, L3
                    # Device 2：L4
            # 朴素流水线（Naive PP）
                # 流程：顺序执行，相邻设备间传递激活值和梯度
                # 通讯：只用点到点（MPI.Send / MPI.Recv）
                    # 不需要 AllReduce
                # 问题：GPU 利用率低（同一时刻只有一张卡在干活）

# 大模型训练的存储消耗
    # 模型状态 (Model States)：必须存的
        # parameters：模型参数
        # gradients：梯度
        # optimizer states：Adam 的 momentum 和 variance
    # 冗余状态 (Residual States)：非必需，训练中额外产生
        # activation：激活值，BWD 时用
        # temporary buffers：临时存储
        # unusable fragment memory：碎片化空间

    # 混合精度训练
        # 为什么：fp32 精度高但计算慢；fp16 计算快但精度低
        # 做法：
            # 存储：fp32 主权重 + momentum + variance
            # 计算（FWD/BWD）：用 fp16
            # 更新：用 fp32

    # 存储计算（以 Φ 为模型参数量）
        # fp32 parameter  = 4Φ
        # fp32 momentum   = 4Φ
        # fp32 variance   = 4Φ
        # fp16 parameter  = 2Φ
        # fp16 gradients  = 2Φ
        # 总计 = 16Φ（K=12 是优化器相关的12，+4是fp16权重和梯度）

# ZERO-DP(Zero Redundancy Optimizer)    
    # 核心思想：用通讯换显存 / 用完即弃，需要再补

    # ZeRO-1 (Pos)：切分优化器状态
        # 做法：每张卡只维护一部分 optimizer states
        # 流程：
            # 1. 各卡算完 FWD+BWD，得到完整梯度
            # 2. 梯度 AllReduce（2Φ）
            # 3. 用各自维护的 OS 更新对应部分的 W
            # 4. 参数 AllGather（Φ），补全 W
        # 显存：(2+2+K/Nd)Φ
        # 通讯量：3Φ（比朴素DP的2Φ多1.5倍）
        # 收益：显存降4倍，通讯增50%

    # ZeRO-2 (Pos + Pg)：再切分梯度
        # 做法：优化器状态 + 梯度 都切分
        # 流程：
            # 1. 各卡算完 FWD+BWD，得到完整梯度
            # 2. 梯度 ReduceScatter（Φ）→ 每卡只拿自己那部分
            # 3. 用各自维护的 O 和 G 更新对应的 W
            # 4. 参数 AllGather（Φ）
        # 显存：(2 + (2+K)/Nd)Φ
        # 通讯量：2Φ（与朴素DP持平）
        # 收益：显存降8倍，通讯不增加（关键突破）

    # ZeRO-3 (Pos + Pg + Pp)：连参数也切分
        # 做法：全部三项都切分
        # 流程：
            # 1. FWD 前：AllGather W（Φ），算完立刻丢弃非自己的
            # 2. BWD 前：再 AllGather W（Φ），算完立刻丢弃
            # 3. BWD 后：梯度 ReduceScatter（Φ）
            # 4. 用自己维护的 O、G 更新自己的 W
        # 显存：(2+2+K)Φ / Nd（几乎无限省显存）
        # 通讯量：3Φ
        # 收益：用1.5倍通讯换60多倍显存

    # ZeRO 本质：模型并行的形式，数据并行的实质
        # 真正的模型并行：各卡只用自己那部分 W 算
        # ZeRO：用的时候拼回完整 W，本质仍是数据并行


