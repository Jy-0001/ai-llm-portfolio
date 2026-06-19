from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown_document = """
# Chapter 1: The Beginning

## Section 1.1: The Old World
This is the story of a time long past.

## Section 1.2: A New Hope
A new hero emerges.

# Chapter 2: The Journey

## Section 2.1: The Call to Adventure
The hero receives a mysterious call.
"""

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(markdown_document)

for split in md_header_splits:
    print(f"Metadata: {split.metadata}")
    print(split.page_content)
    print("-" * 20)
