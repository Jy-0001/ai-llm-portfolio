from langchain_text_splitters import CharacterTextSplitter

sample_text = (
    "LangChain was created by Harrison Chase in October 2022. It provides a framework for developing applications "
    "powered by language models. The library is known for its modularity and ease of use. "
    "One of its key components is the TextSplitter class, which helps in document chunking."
)

text_splitter = CharacterTextSplitter(
    separator=" ",          # 按空格分割
    chunk_size=100,         # 块大小
    chunk_overlap=20,       # 重叠字符数
    length_function=len,
)

docs = text_splitter.create_documents([sample_text])
for i, doc in enumerate(docs):
    print(f"--- Chunk {i+1} ---")
    print(doc.page_content)
