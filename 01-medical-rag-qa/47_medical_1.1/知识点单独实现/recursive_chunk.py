from langchain_text_splitters import RecursiveCharacterTextSplitter

sample_text = (
    "LangChain was created by Harrison Chase in October 2022. \n It provides a framework for developing applications "
    "powered by language models. The library is known for its modularity and ease of use. \n"
    "One of its key components is the TextSplitter class, which helps in document chunking."
)

# 使用与上文相同的 sample_text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    # 默认分隔符为 ["\n\n", "\n", " ", ""]
)

docs = text_splitter.create_documents([sample_text])
for i, doc in enumerate(docs):
    print(f"--- Chunk {i+1} ---")
    print(doc.page_content)
