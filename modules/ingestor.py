"""
modules/ingestor.py
===================
Handles the full RAG ingestion pipeline:
  PDF → Text Extraction → Chunking → Embedding → FAISS Vector Store
"""

import tiktoken
from io import BytesIO
import tempfile, os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def count_tokens(text: str, model: str = "gpt-4") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


def ingest_pdf(uploaded_file, chunk_size: int = 500, chunk_overlap: int = 50) -> dict:
    """
    Full RAG ingestion pipeline.
    Returns dict with: vectorstore, num_pages, num_chunks, total_tokens
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        # STEP 1: Load and parse PDF
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        num_pages = len(pages)

        # STEP 2: Chunk the documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size * 4,
            chunk_overlap=chunk_overlap * 4,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(pages)
        num_chunks = len(chunks)

        # STEP 3: Estimate total tokens
        full_text = " ".join([doc.page_content for doc in chunks])
        total_tokens = count_tokens(full_text)

        # STEP 4: Embed and index with FAISS
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)

    finally:
        os.unlink(tmp_path)

    return {
        "vectorstore": vectorstore,
        "num_pages": num_pages,
        "num_chunks": num_chunks,
        "total_tokens": total_tokens,
    }
