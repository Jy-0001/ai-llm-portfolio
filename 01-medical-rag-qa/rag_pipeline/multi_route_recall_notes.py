# 医疗健康AI Agent项目开发：RAG多路召回 (Agent开发3 / 项目V2.0)
    # 工具：pdfplumber, Milvus, langchain_milvus, FastAPI, DeepSeek, 智谱embedding-3
    # 核心思想：多路召回 (Multi-route Recall) = 不止一个数据源, 不止一种召回方式
        # 第一路(已有)：文本医疗QA数据(jsonl) → Milvus dense+sparse → 内置RRF重排
        # 第二路(新增)：PDF医学教材 → pdfplumber提取 → Milvus + ParentDocumentRetriever
        # 融合：两路结果拼成 context → 喂 DeepSeek 生成
    # 第1步 PDF批处理 (preprocess.py)：把PDF榨成结构化文本
        # 工具：pdfplumber (工业界提取PDF常用)
        # 类：PDFBatchProcessor
            # find_pdf_files     path.glob("**/*.pdf") 递归找所有PDF
            # extract_pdf_content  逐页 extract_text + extract_tables, 单页失败不中断(try/pass)
            # process_batch      批量循环, 每10个文件存一次中间进度
            # _save_results      扁平化 → 两个excel: summary(统计) + detailed_text(逐页全文)
        # 工业级三件套(这节重点不是技术是"生产规范")：
            # 日志：logging 同时写文件 + 控制台, 每篇成功/失败都记
            # 容错：单页提取失败 try/except 跳过, 单篇失败记 error 不崩整批
            # 中间保存：每10篇 dump 一次 progress csv, 防大批量跑一半挂了丢数据
        # 表格提取配置 ADVANCED_TABLE_SETTINGS：vertical/horizontal_strategy="lines" 按线框识别表格
        # 输出：pdf_detailed_text.xlsx (后续 vectors.py 的输入)
    # 第2步 建第二路索引 (vectors.py)：PDF文本灌进 Milvus + 父子检索器
        # Milvus 双索引 (dense + sparse 同库)：
            # dense_index    metric=IP, index=IVF_FLAT      → 向量稠密检索
            # sparse_index   metric=BM25, SPARSE_INVERTED_INDEX → 关键词稀疏检索
            # vector_field=["dense", "sparse"] + BM25BuiltInFunction() → 一个 collection 同时存两种
            # 即"混合检索"在 Milvus 内置层实现, 不再手动拼 EnsembleRetriever
        # Milvus_vector 类(第一路, 文本)：create_vector_store 先建10条再批量插, 用 aadd_documents(异步)
        # Pdf_retriever 类(第二路, PDF)：
            # child_splitter   chunk_size=200, overlap=50, 中文分隔符 ["。","！","？","；","，"...]
            # parent_splitter  chunk_size=1000, overlap=200
            # ParentDocumentRetriever(vectorstore=Milvus, docstore=InMemoryStore)
            # 坑：ParentDocumentRetriever 不支持异步 → 只能 add_documents (同步), 不能 aadd
        # prepare_document       从 jsonl 读, query+"\n"+response 拼成一条 Document (第一路数据)
        # prepare_pdf_document   从 excel 读 text_content, dropna 删空行, 转 Document (第二路数据)
    # 第3步 服务主逻辑 (agent2.py)：FastAPI 两路召回融合
        # 启动时建两个 Milvus 连接：URI=milvus_agent.db(文本) / URI1=pdf_agent.db(PDF)
        # @app.post("/") 单个 query 进来跑两路：
            # 路1：milvus_vectorstore.similarity_search(query, k=10, ranker_type="rrf", ranker_params={"k":100})
            #       Milvus 内置 RRF 重排, 直接出 top-10 → format_docs 拼成 context
            # 路2：parent_retriever.invoke(query) → 取 retrieved_docs[0] 接到 context 后面
        # context(两路拼接) 塞进 SYSTEM_PROMPT + USER_PROMPT → generate_deepseek_answer
        # uvicorn 启动, port=8103, workers=1
    # 关键认知
        # 多路召回的本质：不同数据源/格式各建各的索引, 查询时各跑各的, 结果在 context 层拼起来
        # Milvus vs 之前的 Chroma：这节才真正上生产级向量库, dense+sparse 双索引是 Milvus 原生能力
        # RRF 这次是 Milvus 内置(ranker_type="rrf")，不用自己写 reciprocal_rank_fusion 了
    # 代码里发现的坑(自己用要补)
        # ① docstore 不持久化：vectors.py 灌库时 parent 存在 InMemoryStore, 进程结束就没了;
        #    agent2.py 重启又是空的 InMemoryStore → 第二路 parent_retriever 召回大概率拿不到父文档
        #    (test.py 那个"蜂蜜白醋"答案明显来自第一路文本, 不是PDF教材, 侧面印证第二路没生效)
        #    修复：docstore 换持久化(LocalFileStore / Redis / 数据库), 或索引和服务同进程
        # ② context 类型不一致：路1为空时 context=[] (list), 后面 context + "\n" + res 会 TypeError
        #    修复：else 分支应写 context = "" 而非 []
        # ③ USER_PROMPT 双重格式化：它已是 f-string(填好了), 又调 .format(context, query) 多余,
        #    若 context 里含 { } 字符会崩; 去掉 .format() 即可