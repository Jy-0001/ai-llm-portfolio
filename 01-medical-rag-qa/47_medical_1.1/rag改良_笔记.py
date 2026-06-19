# RAG精确召回策略
    # 基础分块(前面内容已涵盖)：
        # CharacterTextSplitter
        # RecursiveCharacterTextSplitter
    # 结构感知分块
        # 说明：利用文档固有的结构信息 (如标题, 列表, 对话轮次) 作为分块边界, 这种方法逻辑性强, 能更好地保留上下文
    # 智能分块
    # 句子分块



# 医疗健康AI Agent项目开发：RAG精确召回与后处理 (Agent开发2)
    # 工具：LangChain, Chroma, DeepSeek(LLM), 智谱embedding-3
    # 主线：索引(怎么切) → 查询(问得多) → 检索器(找更准) → 后处理(筛干净) → 生成(防)
    # 核心痛点：按字数硬切 → 语义关联丢失
    # 分块策略
        # 基础分块策略：
            # 固定长度分块：CharacterTextSplitter
                # 代码实现：见
                '''fixed_chunk.py'''
            # 递归分块：RecursiveCharacterTextSplitter
                # 代码实现：见
                '''recursive_chunk.py'''
            # 结构感知分块：MarkdownHeaderTextSplitter
                # 代码实现：见 
                '''markdown_chunk.py'''
            # 对话分块
                # 代码实现：见 
                '''dialogue_chunk.py'''
            # 关键超参：
                # chunk_size：256 / 512 / 1024 (按嵌入模型最佳输入选)
                    # 太小→上下文不足; 太大→噪声多, Token贵
                # chunk_overlap：取 chunk_size 的 10%-20%, 防边界切断长句
    # 创造更多检索入口*****
        # QA代理问题："query召回问题" 优于 "query召回陈述"
            # 块 → LLM反向生成代理问题 → 只对代理问题向量化, 挂原文doc_id
            # 命中代理问题 → 通过ID返回原文块
            # → 一个知识点开多个语义入口  (思路同Agent开发1的QA.py)
        # RAG-Fusion：多子查询独立检索 + RRF融合
            # ***重点注意***：和多检索器融合/混合检索的区分*****：
                # 多检索器融合/混合检索：1 query → N retrievers → N ranked lists → RRF
                # 多查询融合/RAG-Fusion：1 query → LLM扩成 N queries → 1 retriever 跑 N 次 → N ranked lists → RRF
            # RRF算法 (Reciprocal Rank Fusion, 倒数排序融合)：
                # 公式：score(doc) = Σ 1 / (k + rank)
                    # rank   doc在某次查询里的排名(0起)
                    # k      平滑常数, 常取60, 削弱头部排名过度影响
                # 直觉：多次查询都靠前的"共识"文档累加分高 → 提到最前
                # 关键：融合"排名"而非"分数", 不同检索器分数尺度不可比
            # 代码实现：见 
            '''rag_fusion.py'''
        # Step-back 后退一步：问太具体时, LLM抽出更高层概括问题
            # 具体+概括 一起检索, 同时拿细节和背景    
            # 代码实现：见
            '''step_back.py'''
        # HyDE 假设性文档嵌入：LLM凭空生成理想答案, 用"假想答案"向量去检索
            # 直觉：假想答案语义无限接近真答案 → 精准"语义导弹"
            # 代码实现：见
            '''HyDE.py'''
        # BM25 稀疏检索：适用于 短文本/特有名词 的查询
            # 代码实现：Agent开发1 已讲
    # 高级RAG召回策略
        # 父文档检索器ParentDocumentRetriever*****：
            # 原理：
                # 索引时: 我们将一份文档切分成许多小的"子文档" (比如单个句子), 并对这些"子文档"进行向量化. 同时, 我们保留一份完整的, 较大的"父文档" (比如整个段落或⻚面).
                # 检索时: 我们用用户查询去匹配那些精细的"子文档" .
                # 返回时: 一旦命中某个"子文档", 我们不返回这个小片段, 而是返回它所属的那个完整的"父文档"作为上下文.

            # 优点：兼具检索的精准度 (匹配小块) 和上下文的完整性 (返回大块)
            # 代码实现：见
            '''parent retriever.py'''
        # EnsembleRetriever 混合检索：稀疏锁"词" + 稠密锁"意"
            # weights加权, 如 [0.4, 0.6] 偏向量语义
            # 见 
            '''parent_retriever.py'''
        # Agentic Chunking 代理分块召回：LLM动态定边界, 输出(标题/自包含文本/代表问题)三件套
            # 使用工具：PydanticOutputParser 强制结构化; 适用高度非结构化文本
            # 代码实现：见 
            '''agentic_chunk.py'''
    # RAG 后处理工程优化：提升上下文信噪比, 痛点是Top-K含噪声且不同等重要
        # 结果精炼：对初步检索到的文档进行重排序, 压缩与筛选, 提升上下文的信噪比
            # Reranker 重排序：Agent开发1 已讲
            # Contextual Compression 上下文压缩：用LLM从每块抽与query直接相关的句子, 丢无关
                # 使用工具：ContextualCompressionRetriever + LLMChainExtractor.from_llm(llm)
                # → 省Token, 帮LLM聚焦
                # 代码实现：见
                '''compression.py'''
        # 架构优化：引入查询路由等模式, 构建更具弹性和智能化的系统
            # Query Routing 查询路由：入口加LLM分类器按意图分发
                # 分支：向量检索 / 摘要 / 结构化过滤 / 不检索直答
                # 实现思路：意图分类Prompt + RunnableBranch    (本课只讲原理, 无代码)
        # 生成控制：通过有效的Prompt工程, 确保语言模型能忠实, 准确地生成回答
        # 系统性防范：优质检索(根本) + 严格Prompt(约束) + 答案核查(后处理) + 强基座(模型)
    # 元数据 & 策略选择
        # 元数据：每块打标签(来源/日期/作者/章节), 精确过滤 → 企业级RAG关键
        # 知识图谱 neo4j：按需上
        # 分块策略选择三步法：
            # 1. 基线   RecursiveCharacterTextSplitter
            # 2. 文档有结构(MD/HTML/代码/对话) → 结构感知分块
            # 3. 精度仍不够 → ParentDocumentRetriever
    # 《易筋经》三认知：
        # 1. 无银弹：按数据特性 & 业务需求迭代
        # 2. 始于简单, 终于复合
        # 3. 分块即建模：怎么分=怎么理解知识