import os
from dotenv import load_dotenv
load_dotenv()
from tqdm import tqdm
import json
import uuid
import time
import pandas as pd
from zai import ZhipuAiClient
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from config import DOCSTORE_PATH, get_logger

logger = get_logger(__name__)


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


class Milvus_vector():
    def __init__(self, client, uri="./milvus_agent.db"):
        self.URI = uri
        self.embeddings = ZhipuAIEmbeddings(client=client)

        # 定义索引类型
        self.dense_index = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
        }
        self.sparse_index = {
            "metric_type": "BM25",
            "index_type": "SPARSE_INVERTED_INDEX"
        }

    def create_vector_store(self, docs):
        init_docs = docs[:10]
        self.vectorstore = Milvus.from_documents(
            documents=init_docs,
            embedding=self.embeddings,
            builtin_function=BM25BuiltInFunction(),  # output_field_names="sparse",
            index_params=[self.dense_index, self.sparse_index],
            vector_field=["dense", "sparse"],
            connection_args={
                "uri": self.URI,
            },
            consistency_level="Bounded",  # 支持 ("Strong", "Session", "Bounded", "Eventually")
            drop_old=False,
        )
        logger.info("已初始化创建 Milvus")

        count = 10
        temp = []
        for doc in tqdm(docs[10:]):
            temp.append(doc)
            if len(temp) >= 5:
                self.vectorstore.aadd_documents(temp)
                count += len(temp)
                temp = []
                print(f"已插入 {count} 条数据......")
                time.sleep(1)

        print(f"总共插入 {count} 条数据......")
        logger.info("已创建 Milvus 索引完成")

        return self.vectorstore


class Pdf_retriever():
    def __init__(self, client, uri="./pdf_agent.db"):
        self.URI = uri
        self.embeddings = ZhipuAIEmbeddings(client=client)

        # 定义索引类型
        self.dense_index = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
        }
        self.sparse_index = {
            "metric_type": "BM25",
            "index_type": "SPARSE_INVERTED_INDEX"
        }

        # 持久化父文档存储（修复重启后父文档丢失的 bug）
        self.docstore = create_kv_docstore(LocalFileStore(DOCSTORE_PATH))

        # 文本分割器
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def create_pdf_vector_store(self, docs):
        self.milvus_vectorstore = Milvus(
            embedding_function=self.embeddings,
            builtin_function=BM25BuiltInFunction(),
            vector_field=["dense", "sparse"],
            index_params=[
                {
                    "metric_type": "IP",
                    "index_type": "IVF_FLAT",
                },
                {
                    "metric_type": "BM25",
                    "index_type": "SPARSE_INVERTED_INDEX"
                }
            ],
            connection_args={"uri": self.URI},
            consistency_level="Bounded",
            drop_old=False,
        )

        # 设置父子文档检索器
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.milvus_vectorstore,
            docstore=self.docstore,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
        )

        # 添加文档
        count = 0
        temp = []
        for doc in tqdm(docs):
            temp.append(doc)
            if len(temp) >= 10:
                # ParentDocumentRetriever()不支持异步等待操作
                self.retriever.add_documents(temp)
                count += len(temp)
                temp = []
                print(f"已插入 {count} 条数据......")
                time.sleep(1)

        print(f"总共插入 {count} 条数据......")
        logger.info("基于PDF文档数据的 Milvus 索引完成")

        return self.retriever


# 插入数据
def prepare_document(file_path=['./data/data.jsonl', './data/train.jsonl']):
    # 逐条取出文本数据, 创建嵌入张量, 然后将张量数据插入Milvus
    file_path1 = file_path[0]

    count = 0
    docs = []
    with open(file_path1, 'r', encoding='utf-8') as f:
        for line in f:
            content = json.loads(line.strip())
            prompt = content['query'] + "\n" + content['response']

            temp_doc = Document(page_content=prompt, metadata={"doc_id": str(uuid.uuid4())})
            docs.append(temp_doc)
            count += 1

    logger.info("已加载 %d 条数据", count)

    return docs


def prepare_pdf_document(file_path="./pdf_output/pdf_detailed_text.xlsx"):
    df = pd.read_excel(file_path)

    # 空行直接删除, 否则后续处理报错
    df = df.dropna(subset=['text_content'])

    # 将DataFrame转换为LangChain文档
    documents = []
    for _, row in df.iterrows():
        # 确保 text_content 是字符串, 且不为 NaN
        text_content = str(row['text_content']) if pd.notna(row['text_content']) else ""

        doc = Document(
            page_content=text_content.strip(),
            metadata={"doc_id": str(uuid.uuid4())}
        )
        documents.append(doc)

    print(f"成功加载 {len(documents)} 个文档")

    return documents


if __name__ == "__main__":
    '''
    # 预处理即将插入 Milvus 的文档数据
    docs = prepare_document()
    print("预处理文档数据成功......")

    # 创建 Milvus 连接
    milvus_vectorstore = Milvus_vector(client)
    print("创建Milvus连接成功......")

    # 创建向量索引
    vectorstore = milvus_vectorstore.create_vector_store(docs)
    '''

    # 将 PDF 后处理文档中的数据, 封装成Document
    docs = prepare_pdf_document()
    print("预处理 PDF 文档数据成功......")
    # print(docs[0])

    pdf_vectorstore = Pdf_retriever(client)
    print("创建 PDF Milvus 连接成功......")

    retriever = pdf_vectorstore.create_pdf_vector_store(docs)
    print("创建基于 Milvus 数据库的父子文档检索器成功......")
    print(retriever)

    print("全部初始化完成, 可以开始问答了......")