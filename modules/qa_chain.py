"""
modules/qa_chain.py
===================
Implements RAG-based Q&A using ConversationalRetrievalChain.
"""

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

# langchain-classic provides the legacy chains/prompts API
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


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
{context}"""

FINANCIAL_QA_HUMAN = """Question: {question}

Answer (cite page numbers where possible):"""


def build_qa_chain(vectorstore: FAISS, top_k: int = 3, memory=None) -> ConversationalRetrievalChain:
    """Build a ConversationalRetrievalChain for financial document Q&A."""
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1, max_tokens=600)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )

    system_message = SystemMessagePromptTemplate(
        prompt=PromptTemplate(template=FINANCIAL_QA_SYSTEM, input_variables=["context"])
    )
    human_message = HumanMessagePromptTemplate(
        prompt=PromptTemplate(template=FINANCIAL_QA_HUMAN, input_variables=["question"])
    )
    qa_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        verbose=False
    )
    return chain


def ask_question(chain: ConversationalRetrievalChain, question: str, memory=None) -> dict:
    """Run a question through the Q&A chain and return answer + sources."""
    try:
        result = chain.invoke({"question": question})
        answer = result.get("answer", "I couldn't generate an answer.")

        source_docs = result.get("source_documents", [])
        source_pages = []
        for doc in source_docs:
            page = doc.metadata.get("page")
            if page is not None and page + 1 not in source_pages:
                source_pages.append(page + 1)

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
