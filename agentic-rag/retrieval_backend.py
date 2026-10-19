import argparse
import os
import hashlib
import json
import uuid
from pathlib import Path
import pymupdf4llm
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient, DataType

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")
DOCS_DIR = BASE_DIR / "docs"
MARKDOWN_DIR = BASE_DIR / "markdown_docs"
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL")
MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
# 使用独立集合，后续重建只影响此阶段的索引。
COLLECTION = "stage2_dense_chunks"
EMBEDDING_MODEL = "quentinz/bge-large-zh-v1.5:latest"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 60

# 本阶段以字符计量，不把 300 字符误当成 300 token。
# 索引签名同时记录模型和切分参数，避免查询悄悄使用另一套配置。
INDEX_SIGNATURE = json.dumps(
    {"embedding": EMBEDDING_MODEL, "chunk_size": CHUNK_SIZE,
     "chunk_overlap": CHUNK_OVERLAP, "pipeline": "stage2-v1"},
    sort_keys=True,
)


def load_pdf_documents() -> list[Document]:
    pdf_paths = sorted(p for p in DOCS_DIR.rglob("*")
                        if p.is_file() and p.suffix.lower() == ".pdf")
    if not pdf_paths:
        # 没有输入就退出，不能拿一个空输入去重建已有集合。
        raise ValueError(f"请先向 {DOCS_DIR} 放入文字可选中的 PDF。")

    documents = []
    for pdf_path in pdf_paths:
        relative = pdf_path.relative_to(DOCS_DIR)
        # 保留 .pdf 再追加 .md，避免 A.pdf 和 A.PDF 转换产物同名。
        markdown_path = MARKDOWN_DIR / relative.parent / (relative.name + ".md")
        markdown_path.parent.mkdir(parents=True, exist_ok=True)

        # 每次重建都重新转换，避免源 PDF 修改后仍读取旧 Markdown。
        # ignore_images 适合此阶段的文字检索，不代表支持图表理解。
        markdown = pymupdf4llm.to_markdown(str(pdf_path), ignore_images=True)
        if not markdown.strip():
            raise ValueError(f"转换结果为空，请检查是否为扫描件：{relative}")
        markdown_path.write_text(markdown, encoding="utf-8")

        documents.append(Document(
            page_content=markdown,
            metadata={
                "source": relative.as_posix(),
                "markdown_source": markdown_path.relative_to(BASE_DIR).as_posix(),
                # 原文件摘要用于核对来源版本，不承担增量索引功能。
                "source_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
            },
        ))
    return documents

