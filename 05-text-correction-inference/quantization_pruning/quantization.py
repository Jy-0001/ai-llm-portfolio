#模型量化：
    #本质：将模型参数从高精度浮点数压缩为低比特整数
        #float32（32位浮点）→ int8（8位有符号整数）
        #float32：占4字节，表示范围±3.4×10³⁸，精度极高
        #int8：占1字节，表示范围-128~127
        #换算公式（线性量化）：int_val = round(float_val / scale + zero_point)
            #scale 和 zero_point 是每层推导出的量化参数
        #代价：引入量化误差，精度略有损失
        #收益：显存/磁盘降至原来约1/4，推理速度提升（整数运算比浮点更快）

    #量化的三种方式：
        #动态量化：权重提前量化为int8，激活值在推理时动态计算scale
            #适合NLP模型（计算瓶颈在Linear层的矩阵乘法）
        #静态量化：需要校准集，激活值的scale提前统计好
        #量化感知训练（QAT）：训练阶段模拟量化误差，精度损失最小

    #pytorch的动态量化：
        #前提：PyTorch版本 >= 1.3.0
        #做法：直接使用torch.quantization.quantize_dynamic()实现量化
            #model：原始模型
            #{torch.nn.Linear}：指定量化哪些层（NLP一般只量化Linear）
            #dtype=torch.qint8：目标精度为8位有符号整数
        #示例：
            #quantized_model = torch.quantization.quantize_dynamic(
            #    model, {torch.nn.Linear}, dtype=torch.qint8
            #)

    #bert量化：
        #步骤1：对模型进行动态量化并评估
""" 见run.py 最后部分"""
            #量化后在验证集跑一次，对比量化前后的acc/f1
            #目的：确认精度损失在可接受范围内（通常<1%）
        #步骤2：对比模型压缩后的大小
            #BERT初始模型（bert_model.bin）大小：392.51MB
            #BERT量化模型（bert_model_quantized.bin）大小：145.58MB
            #压缩比约2.7x，而非理论上的4x，原因：
                #Embedding层通常不参与量化（词表大但量化收益低）
                #LayerNorm、bias保持float32以保证数值稳定性
                #文件中还包含scale和zero_point本身的存储开销
