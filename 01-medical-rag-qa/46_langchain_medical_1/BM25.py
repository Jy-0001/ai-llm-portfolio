import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.stores import InMemoryStore
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
# 示例: 使用 EnsembleRetriever 实现混合搜索
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import uuid
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from zai import ZhipuAiClient
from langchain.embeddings.base import Embeddings
import os
import time

# 嵌入模型, 采用清华智谱最新的embedding-3, 实例化智谱client对象
client = ZhipuAiClient(api_key=os.getenv("ZHIPU_API_KEY"))

class ZhipuAIEmbeddings(Embeddings):
    def __init__(self, client):
        self.client = client

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            # 调用清华智谱最新版本的 embeddings 方法
            response = self.client.embeddings.create(
                model="embedding-3",
                input=[text],
            )
            embeddings.append(response.data[0].embedding)
        return embeddings

    def embed_query(self, text):
        # 查询文档
        return self.embed_documents([text])[0]

embeddings = ZhipuAIEmbeddings(client=client)

# 自己申请DeepSeek的api_key, 并充实付费

# 从环境变量获取 DeepSeek API Key
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

# 初始化 DeepSeek 模型
llm = ChatOpenAI(
    model="deepseek-chat",  # 或者使用 "deepseek-reasoner"
    openai_api_key=deepseek_api_key,  # 你的 DeepSeek API 密钥
    base_url="https://api.deepseek.com/v1",  # DeepSeek 的 API 端点
    temperature=0.7,  # 控制创造性, 根据需求调整
    max_tokens=2048,  # 根据模型最大上下文窗口调整
)

docs = [
    Document(page_content="爸爸前几天去医院刚检查出有糖尿病,今年60岁血糖10请问专家吃什么能好转?"
             "求各位告诉我一下谢谢? "
             "医生回复:首先，建议您的父亲遵循医生的建议，按时服药和定期检查血糖。其次，饮食对于糖尿"
             "病患者非常重要，建议他遵循以下饮食原则：\n "
             "1. 控制碳水化合物的摄入量，尤其是高糖、高淀粉的食物，例如白米饭、面条、面包、糖果、饮"
             "料等。\n "
             "2. 增加蔬菜和水果的摄入量，尤其是低糖、低淀粉的蔬菜，例如菠菜、芦笋、西兰花、豆腐等，"
             "以及水果中的柚子、苹果、草莓、蓝莓等。\n "
             "3. 适量增加优质蛋白质的摄入量，例如鸡蛋、鱼、豆类、瘦肉等。\n "
             "4. 避免食用高脂、高盐的食物，例如猪肉、牛肉、奶油、蛋糕等。\n "
             "5. 控制饮食量和饮食频率，避免大量进食和过度饥饿。\n "
             "6. 坚持规律的饮食习惯，每日三餐定时定量，避免随意进食和过度饥饿。\n "
             "此外，适量运动也有助于控制血糖，建议您的父亲每天进行适当的运动，例如散步、慢跑、瑜伽"
             "等。最后，建议您的父亲定期复查血糖，及时调整治疗方案。",
             metadata={"doc_id": str(uuid.uuid4())}),
    Document(page_content="不小心得了牛皮癣了，但是不知道有什么特点。感觉和湿疹差不多。都很痒，反"
             "反复复的。一直好不了。"
             "牛皮癣和湿疹都是常见的皮肤病，但两者的病因和治疗方法不同。牛皮癣是一种自身免疫性疾病，"
             "主要表现为皮肤出现红斑、鳞屑、瘙痒、疼痛等症状。"
             "而湿疹则是一种非传染性皮肤炎症，表现为皮肤红肿、瘙痒、渗出、结痂等症状。\n "
             "如果您怀疑自己得了牛皮癣，建议尽快就医。医生会根据病情给出相应的治疗方案，包括外用药、"
             "口服药、光疗等。"
             "同时，保持皮肤清洁、避免刺激、保持心情舒畅也有助于缓解症状。",
             metadata={"doc_id": str(uuid.uuid4())}),
]

doc_ids = [doc.metadata["doc_id"] for doc in docs]
print('doc_ids:', doc_ids)

question_gen_prompt_str = (
    "你是一位AI医学专家. 请根据以下文档内容, 生成3个用户可能会提出的, 高度相关的问题.\n"
    "只返回问题列表, 每个问题占一行, 不要有其他前缀或编号.\n"
    "文档内容:\n"
    "----------\n"
    "{content}\n"
    "----------\n"
)

question_gen_prompt = ChatPromptTemplate.from_template(question_gen_prompt_str)
question_generator_chain = question_gen_prompt | llm | StrOutputParser()

sub_docs = []
for i, doc in enumerate(docs):
    doc_id = doc_ids[i]
    generated_questions = question_generator_chain.invoke({"content": doc.page_content}).split("\n")
    generated_questions = [q.strip() for q in generated_questions if q.strip()]
    for q in generated_questions:
        sub_docs.append(Document(page_content=q, metadata={"doc_id": doc_id}))

print("创建Chroma向量数据库, 并添加文档...")
vectorstore = Chroma.from_documents(documents=sub_docs, embedding=embeddings)

# 假设 all_splits 和 vectorstore 已准备好
# 初始化关键词检索器 (Sparse Retriever)
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 3  # 检索3个结果

# 初始化向量检索器 (Dense Retriever)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 初始化 EnsembleRetriever, 并设置权重
# weights参数决定了最终排序时, 两种检索器结果的权重
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]  # 稍微偏重向量搜索的语义理解能力
)

query = "糖尿病患者有什么饮食建议?"
start_time = time.time()
retrieved_docs = ensemble_retriever.invoke(query)
end_time = time.time()

print(f"混合搜索召回了 {len(retrieved_docs)} 个文档。")

for doc in retrieved_docs:
    print(doc.page_content)
    print('-' * 70)

print(f"检索耗时:{end_time - start_time:.2f}秒")