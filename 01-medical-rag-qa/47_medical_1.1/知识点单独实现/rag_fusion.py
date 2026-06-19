import os
from dotenv import load_dotenv
load_dotenv()
import os
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import uuid
import operator
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from zai import ZhipuAiClient
from langchain.embeddings.base import Embeddings

# 实例化智谱client对象
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

# 从环境变量获取 DeepSeek API Key
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
# 初始化 DeepSeek 模型
llm = ChatOpenAI(
    model="deepseek-chat",                      # 或者使用 "deepseek-reasoner"
    openai_api_key=deepseek_api_key,            # 你的 DeepSeek API 密钥
    base_url="https://api.deepseek.com/v1",     # DeepSeek 的 API 端点
    temperature=0.7,                            # 控制创造性, 根据需求调整
    max_tokens=2048,                            # 根据模型最大上下文窗口调整
)

docs = [
    Document(page_content="爸爸前几天去医院刚检查出有糖尿病,今年60岁血糖10请问专家吃什么能好转?求各位告诉我一下谢谢? \
医生回复:首先，建议您的父亲遵循医生的建议，按时服药和定期检查血糖。其次，饮食对于糖尿病患者非常重要，建议他遵循以下饮食原则：\n \
 1. 控制碳水化合物的摄入量，尤其是高糖、高淀粉的食物，例如白米饭、面条、面包、糖果、饮料等。\n \
 2. 增加蔬菜和水果的摄入量，尤其是低糖、低淀粉的蔬菜，例如菠菜、芦笋、西兰花、豆腐等，以及水果中的柚子、苹果、草莓、蓝莓等。\n \
 3. 适量增加优质蛋白质的摄入量，例如鸡蛋、鱼、豆类、瘦肉等。\n \
 4. 避免食用高脂、高盐的食物，例如猪肉、牛肉、奶油、蛋糕等。\n \
 5. 控制饮食量和饮食频率，避免大量进食和过度饥饿。\n \
 6. 坚持规律的饮食习惯，每日三餐定时定量，避免随意进食和过度饥饿。\n \
此外，适量运动也有助于控制血糖，建议您的父亲每天进行适当的运动，例如散步、慢跑、瑜伽等。最后，建议您的父亲定期复查血糖，及时调整治疗方案。", \
metadata={"doc_id": str(uuid.uuid4())}),
    Document(page_content="不小心得了牛皮癣了，但是不知道有什么特点。感觉和湿疹差不多。都很痒，反反复复的。一直好不了。\
牛皮癣和湿疹都是常见的皮肤病，但两者的病因和治疗方法不同。牛皮癣是一种自身免疫性疾病，主要表现为皮肤出现红斑、鳞屑、瘙痒、疼痛等症状。\
而湿疹则是一种非传染性皮肤炎症，表现为皮肤红肿、瘙痒、渗出、结痂等症状。\n \
如果您怀疑自己得了牛皮癣，建议尽快就医。医生会根据病情给出相应的治疗方案，包括外用药、口服药、光疗等。\
同时，保持皮肤清洁、避免刺激、保持心情舒畅也有助于缓解症状。", \
metadata={"doc_id": str(uuid.uuid4())}),
]

doc_ids = [doc.metadata["doc_id"] for doc in docs]
print('doc_ids:', doc_ids)

question_gen_prompt_str = (
    "你是一位AI医学专家。请根据以下文档内容,生成3个用户可能会提出的,高度相关的问题。\n"
    "只返回问题列表，每个问题占一行，不要有其他前缀或编号。\n"
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

print('sub_docs:', sub_docs)
print("创建Chroma向量数据库, 并添加文档...")
vectorstore = Chroma.from_documents(documents=sub_docs, embedding=embeddings)

# 定义一个用于生成子查询的链 (Chain)
query_gen_prompt = ChatPromptTemplate.from_messages([
    ("user", "你是一位AI医学专家。请根据以下问题, 生成3个不同角度的, 语义相似的查询问题，注意是问题。\n"
             "每个查询占一行, 不要有其他前缀或编号。\n"
             "原始问题: {original_question}")
])
generate_queries_chain = query_gen_prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))


# 定义 RRF 算法函数
def reciprocal_rank_fusion(retrieval_results: list[list[Document]], k=60):
    fused_scores = {}
    for doc_list in retrieval_results:
        for rank, doc in enumerate(doc_list):
            doc_id = doc.page_content
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
            fused_scores[doc_id] += 1 / (k + rank)

    # print('\nretrieval_results:', retrieval_results,'\n')
    # print('\nfused_scores:', fused_scores,'\n')
    reranked_results = [
        next((doc for doc_list in retrieval_results for doc in doc_list if doc.page_content == doc_id), None)
        for doc_id, score in sorted(fused_scores.items(), key=operator.itemgetter(1), reverse=True)
    ]

    return [doc for doc in reranked_results if doc is not None]


def rag_fusion_pipeline(original_question: str):
    # 生成多个查询
    generated_queries = generate_queries_chain.invoke({"original_question": original_question})
    all_queries = [original_question] + generated_queries
    print(f"生成的查询: {all_queries}")

    # 独立检索每个查询
    retriever = vectorstore.as_retriever()
    retrieval_results = [retriever.invoke(q) for q in all_queries]

    # 应用RRF算法对结果进行融合和重排
    final_docs = reciprocal_rank_fusion(retrieval_results)
    return final_docs


user_query = "糖尿病患者有什么饮食建议?"
fusion_docs = rag_fusion_pipeline(user_query)
print(fusion_docs, f"RAG-Fusion 对 '{user_query}' 的检索结果")
