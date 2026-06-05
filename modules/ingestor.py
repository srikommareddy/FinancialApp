"""
modules/ingestor.py
===================
Handles the full RAG ingestion pipeline:
  PDF → Text Extraction → Chunking → Embedding → FAISS Vector Store

Learning Points:
  - PyPDFLoader for PDF text extraction
  - RecursiveCharacterTextSplitter for semantic chunking
  - OpenAIEmbeddings (text-embedding-ada-002) for dense vectors
  - FAISS.from_documents to build in-memory vector index
"""

import tiktoken
from io import BytesIO
import tempfile, os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Estimate token count using tiktoken."""
    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        # Rough fallback: 1 token ≈ 4 characters
        return len(text) // 4


def ingest_pdf(uploaded_file, chunk_size: int = 500, chunk_overlap: int = 50) -> dict:
    """
    Full RAG ingestion pipeline.

    Args:
        uploaded_file: Streamlit UploadedFile object (PDF)
        chunk_size: Target token size per chunk
        chunk_overlap: Overlap tokens between adjacent chunks

    Returns:
        dict with keys: vectorstore, num_pages, num_chunks, total_tokens
    """

    # ── STEP 1: Save uploaded file to a temp path ───────────────────────────
    # PyPDFLoader requires a file path, not a file object
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        # ── STEP 2: Load and parse PDF ──────────────────────────────────────
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()          # Returns List[Document], one per page
        num_pages = len(pages)

        # ── STEP 3: Chunk the documents ─────────────────────────────────────
        # RecursiveCharacterTextSplitter tries to split on:
        # ["\n\n", "\n", " ", ""] in order — preserving semantic boundaries
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size * 4,    # approximate chars from token count
            chunk_overlap=chunk_overlap * 4,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(pages)
        num_chunks = len(chunks)

        # ── STEP 4: Estimate total tokens ───────────────────────────────────
        full_text = " ".join([doc.page_content for doc in chunks])
        total_tokens = count_tokens(full_text)

        # ── STEP 5: Embed and index with FAISS ──────────────────────────────
        # OpenAIEmbeddings uses text-embedding-ada-002 by default
        # Produces 1536-dimensional dense vectors
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)

    finally:
        os.unlink(tmp_path)   # Clean up temp file

    return {
        "vectorstore": vectorstore,
        "num_pages": num_pages,
        "num_chunks": num_chunks,
        "total_tokens": total_tokens,
    }
