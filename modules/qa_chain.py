"""
modules/qa_chain.py
===================
Implements RAG-based Q&A using ConversationalRetrievalChain.

Flow per query:
  1. Condense question: Rewrite user question using chat history for standalone context
  2. Retrieve: Top-K similar chunks from FAISS
  3. Generate: GPT-4 answers using retrieved context + system prompt
  4. Return: Answer + source page references

Learning Points:
  - ConversationalRetrievalChain (handles multi-turn context)
  - Custom financial system prompt via combine_docs_chain_kwargs
  - Source document tracking for citation
  - Memory integration for conversation history
"""

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.vectorstores import FAISS


# ── System Prompt for Q&A ──────────────────────────────────────────────────
FINANCIAL_QA_SYSTEM = """You are FinSight AI, an expert financial analyst assistant.

Your role is to answer questions about financial documents with precision and clarity.

RULES:
1. ONLY answer based on the provided document context below.
2. If the answer is not in the context, say "I couldn't find this information in the document."
3. Always cite specific figures, dates, and page references when available.
4. Use professional financial language.
5. If a question requires calculation, show your work step by step.
6. Never speculate beyond what the document states.

CONTEXT FROM DOCUMENT:
{context}

Answer the user's question based strictly on the above context."""

FINANCIAL_QA_HUMAN = """Question: {question}

Answer (cite page numbers where possible):"""


def build_qa_chain(
    vectorstore: FAISS,
    top_k: int = 3,
    memory=None
) -> ConversationalRetrievalChain:
    """
    Build a ConversationalRetrievalChain for financial document Q&A.

    Args:
        vectorstore: Indexed FAISS vectorstore
        top_k: Number of chunks to retrieve per query
        memory: ConversationBufferWindowMemory instance

    Returns:
        Configured ConversationalRetrievalChain
    """
    # ── LLM ───────────────────────────────────────────────────────────────
    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0.1,    # Very low temp for factual Q&A
        max_tokens=600
    )

    # ── Retriever from FAISS ──────────────────────────────────────────────
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )

    # ── Custom QA Prompt ──────────────────────────────────────────────────
    system_message = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            template=FINANCIAL_QA_SYSTEM,
            input_variables=["context"]
        )
    )
    human_message = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template=FINANCIAL_QA_HUMAN,
            input_variables=["question"]
        )
    )
    qa_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    # ── Build Chain ───────────────────────────────────────────────────────
    # ConversationalRetrievalChain:
    #   Step 1 → Condense question (uses chat history to make it standalone)
    #   Step 2 → Retrieve top-K chunks
    #   Step 3 → Generate answer with QA prompt
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,       # We need this for citations
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        verbose=False
    )

    return chain


def ask_question(chain: ConversationalRetrievalChain, question: str, memory=None) -> dict:
    """
    Run a question through the Q&A chain and return answer + sources.

    Args:
        chain: Built ConversationalRetrievalChain
        question: User's question string
        memory: Memory object (used to retrieve chat history)

    Returns:
        dict with: answer, source_pages, source_texts
    """
    try:
        result = chain.invoke({"question": question})
        answer = result.get("answer", "I couldn't generate an answer.")

        # Extract source page numbers from retrieved documents
        source_docs = result.get("source_documents", [])
        source_pages = []
        for doc in source_docs:
            page = doc.metadata.get("page")
            if page is not None and page + 1 not in source_pages:
                source_pages.append(page + 1)   # Convert 0-indexed to 1-indexed

        return {
            "answer": answer,
            "source_pages": sorted(source_pages),
            "source_texts": [doc.page_content[:200] for doc in source_docs]
        }

    except Exception as e:
        return {
            "answer": f"Error generating answer: {str(e)}",
            "source_pages": [],
            "source_texts": []
        }
