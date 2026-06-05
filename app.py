"""
FinSight AI — Intelligent Financial Document Summarizer & Q&A Bot
=================================================================
Capstone Project for NVIDIA Gen AI LLM Associate Certification
Demonstrates: RAG Pipeline · Abstractive Summarization · Guardrails · Memory
"""

import os
import streamlit as st

from modules.ingestor import ingest_pdf
from modules.summarizer import generate_summary
from modules.qa_chain import build_qa_chain, ask_question
from modules.guardrails import is_financial_query, inject_disclaimer
from modules.memory import init_memory, add_to_memory, get_history

# ── Secrets: st.secrets on Cloud; env var locally ──────────────────────────
def _load_api_key() -> str:
    try:
        return st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.environ.get("OPENAI_API_KEY", "")

_env_key = _load_api_key()
if _env_key:
    os.environ["OPENAI_API_KEY"] = _env_key

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSight AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .stApp { background: #0A0F1E; color: #E8EAF6; }

    .main-header {
        background: linear-gradient(135deg, #0D47A1 0%, #006064 100%);
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #1565C0;
    }
    .main-header h1 {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.2rem;
        font-weight: 600;
        color: #FFFFFF;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #B3E5FC;
        margin: 0.4rem 0 0 0;
        font-size: 0.95rem;
        font-weight: 300;
    }

    .metric-card {
        background: #111827;
        border: 1px solid #1E3A5F;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .metric-card .label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #64B5F6;
        margin-bottom: 0.3rem;
        font-family: 'IBM Plex Mono', monospace;
    }
    .metric-card .value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #FFFFFF;
    }

    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #00ACC1;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #1E3A5F;
    }

    .summary-box {
        background: #0D1B2E;
        border: 1px solid #1565C0;
        border-left: 4px solid #00ACC1;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        font-size: 0.92rem;
        line-height: 1.7;
        color: #CFD8DC;
    }

    .chat-user {
        background: #1A237E;
        border-radius: 12px 12px 4px 12px;
        padding: 0.8rem 1.1rem;
        margin: 0.6rem 0;
        color: #E8EAF6;
        font-size: 0.9rem;
        max-width: 80%;
        margin-left: auto;
    }
    .chat-bot {
        background: #0D1B2E;
        border: 1px solid #1E3A5F;
        border-radius: 12px 12px 12px 4px;
        padding: 0.8rem 1.1rem;
        margin: 0.6rem 0;
        color: #CFD8DC;
        font-size: 0.9rem;
        max-width: 85%;
    }
    .chat-bot .source-badge {
        display: inline-block;
        background: #0D47A1;
        color: #90CAF9;
        font-size: 0.68rem;
        padding: 2px 8px;
        border-radius: 20px;
        margin-top: 0.5rem;
        font-family: 'IBM Plex Mono', monospace;
    }

    .guardrail-warning {
        background: #1A0A00;
        border: 1px solid #BF360C;
        border-left: 4px solid #FF6D00;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        color: #FFCCBC;
        font-size: 0.85rem;
        margin: 0.5rem 0;
    }
    .disclaimer {
        background: #0A1500;
        border: 1px solid #33691E;
        border-radius: 6px;
        padding: 0.5rem 0.8rem;
        color: #DCEDC8;
        font-size: 0.75rem;
        margin-top: 0.5rem;
        font-style: italic;
    }

    .chunk-info {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: #546E7A;
        margin-top: 0.3rem;
    }

    .stButton>button {
        background: linear-gradient(135deg, #1565C0, #006064);
        color: white;
        border: none;
        border-radius: 8px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
        letter-spacing: 0.5px;
        padding: 0.6rem 1.5rem;
        transition: opacity 0.2s;
        width: 100%;
    }
    .stButton>button:hover { opacity: 0.85; }

    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background: #111827 !important;
        color: #E8EAF6 !important;
        border: 1px solid #1E3A5F !important;
        border-radius: 8px !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
    }
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: #00ACC1 !important;
        box-shadow: 0 0 0 2px rgba(0,172,193,0.2) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: #111827;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        letter-spacing: 0.5px;
        color: #78909C;
        background: transparent;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
    }
    .stTabs [aria-selected="true"] {
        background: #1565C0 !important;
        color: white !important;
    }

    .pipeline-step {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        padding: 0.6rem 0.8rem;
        background: #111827;
        border-radius: 8px;
        margin-bottom: 0.4rem;
        border-left: 3px solid #1565C0;
        font-size: 0.82rem;
        color: #B0BEC5;
    }
    .pipeline-step .step-num {
        background: #1565C0;
        color: white;
        width: 22px;
        height: 22px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.7rem;
        font-weight: 700;
        flex-shrink: 0;
        font-family: 'IBM Plex Mono', monospace;
    }

    div[data-testid="stSidebar"] { background: #080D1A; border-right: 1px solid #1E3A5F; }
    div[data-testid="stSidebar"] h2 { color: #64B5F6; font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1.5px; }
    .stFileUploader { background: #111827; border: 1px dashed #1E3A5F; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Session State Init ───────────────────────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "doc_metadata" not in st.session_state:
    st.session_state.doc_metadata = {}
if "memory" not in st.session_state:
    st.session_state.memory = init_memory()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📊 FinSight AI</h1>
    <p>Intelligent Financial Document Summarizer &amp; Q&amp;A — Powered by RAG + LangChain</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙ Configuration")

    _key_from_secrets = bool(os.environ.get("OPENAI_API_KEY"))
    if _key_from_secrets:
        st.success("✅ API key loaded from Secrets", icon="🔐")
        api_key = os.environ["OPENAI_API_KEY"]
    else:
        api_key = st.text_input("OpenAI API Key", type="password",
                                 help="Enter your OpenAI API key (or add to Streamlit Secrets)")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

    st.markdown("---")
    st.markdown("## 📄 Document Upload")

    uploaded_file = st.file_uploader(
        "Upload Financial Document",
        type=["pdf"],
        help="Upload a PDF: Annual Report, 10-K, Earnings Release, etc."
    )

    chunk_size = st.slider("Chunk Size (tokens)", 200, 1000, 500, 50,
                            help="Size of each document chunk for RAG indexing")
    chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50, 10,
                               help="Overlap between adjacent chunks to preserve context")
    top_k = st.slider("Top-K Retrieval", 1, 8, 3,
                       help="Number of chunks retrieved per query")

    st.markdown("---")
    st.markdown("## 🛡 Guardrails")
    guardrails_on = st.toggle("Enable Financial Guardrails", value=True,
                               help="Block non-financial queries; inject disclaimer")
    disclaimer_on = st.toggle("Auto-append Disclaimer", value=True,
                               help="Add standard financial disclaimer to answers")

    st.markdown("---")
    st.markdown("## 📊 RAG Pipeline")
    for i, step in enumerate([
        "PDF Text Extraction",
        "Semantic Chunking",
        "Embedding (text-embedding-ada-002)",
        "FAISS Vector Indexing",
        "Query → Top-K Retrieval",
        "GPT-4 Generation"
    ], 1):
        st.markdown(f"""
        <div class="pipeline-step">
            <div class="step-num">{i}</div>
            {step}
        </div>""", unsafe_allow_html=True)

    if st.session_state.doc_metadata:
        st.markdown("---")
        st.markdown("## 📋 Document Info")
        meta = st.session_state.doc_metadata
        for k, v in meta.items():
            st.markdown(f"""<div class="metric-card">
                <div class="label">{k}</div>
                <div class="value">{v}</div>
            </div>""", unsafe_allow_html=True)

# ── Document Processing ───────────────────────────────────────────────────────
if uploaded_file and api_key:
    if st.session_state.vectorstore is None or \
       st.session_state.doc_metadata.get("Filename") != uploaded_file.name:

        with st.spinner("🔄 Ingesting document — chunking, embedding, indexing..."):
            result = ingest_pdf(
                uploaded_file,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            st.session_state.vectorstore = result["vectorstore"]
            st.session_state.doc_metadata = {
                "Filename": uploaded_file.name,
                "Pages": str(result["num_pages"]),
                "Chunks": str(result["num_chunks"]),
                "Tokens (est.)": f"{result['total_tokens']:,}"
            }
            st.session_state.qa_chain = build_qa_chain(
                st.session_state.vectorstore,
                top_k=top_k,
                memory=st.session_state.memory
            )
            st.session_state.summary = None  # reset summary for new doc
            st.session_state.chat_history = []
            st.session_state.memory = init_memory()
        st.success(f"✅ Document indexed — {result['num_chunks']} chunks across {result['num_pages']} pages")

elif uploaded_file and not api_key:
    st.warning("⚠️ Please enter your OpenAI API key in the sidebar to proceed.")

# ── Main Tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📝  SUMMARIZER", "💬  Q&A CHAT", "🔬  ARCHITECTURE"])

# ── TAB 1: SUMMARIZER ─────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Document Summarization</div>', unsafe_allow_html=True)

    if st.session_state.vectorstore is None:
        st.info("📤 Upload a financial PDF document in the sidebar to begin.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            summary_type = st.selectbox("Summary Type", [
                "Executive Summary (2-3 paragraphs)",
                "Bullet-Point Key Findings",
                "Risk Factors Focus",
                "Financial Metrics Focus",
                "Earnings Highlights"
            ])
        with col2:
            summary_length = st.selectbox("Length", ["Concise (150 words)", "Standard (300 words)", "Detailed (500 words)"])
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            summarize_btn = st.button("⚡ Generate Summary")

        if summarize_btn:
            with st.spinner("🧠 Analyzing document with Map-Reduce summarization..."):
                summary = generate_summary(
                    st.session_state.vectorstore,
                    summary_type=summary_type,
                    length=summary_length
                )
                st.session_state.summary = summary

        if st.session_state.summary:
            st.markdown('<div class="section-header">Generated Summary</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="summary-box">{st.session_state.summary}</div>',
                        unsafe_allow_html=True)
            if disclaimer_on:
                st.markdown('<div class="disclaimer">⚠️ This summary is AI-generated from the uploaded document for informational purposes only. It does not constitute financial advice. Always consult a qualified financial advisor before making investment decisions.</div>', unsafe_allow_html=True)
            st.download_button(
                "⬇️ Download Summary",
                data=st.session_state.summary,
                file_name="finsight_summary.txt",
                mime="text/plain"
            )

# ── TAB 2: Q&A CHAT ────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Financial Document Q&A</div>', unsafe_allow_html=True)

    if st.session_state.vectorstore is None:
        st.info("📤 Upload a financial PDF document in the sidebar to begin.")
    else:
        # Chat history display
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>',
                                unsafe_allow_html=True)
                else:
                    sources_html = ""
                    if msg.get("sources"):
                        badges = "".join([f'<span class="source-badge">📄 Page {s}</span> '
                                          for s in msg["sources"]])
                        sources_html = f"<div style='margin-top:0.5rem'>{badges}</div>"
                    st.markdown(
                        f'<div class="chat-bot">🤖 {msg["content"]}{sources_html}</div>',
                        unsafe_allow_html=True
                    )
                    if msg.get("guardrail_blocked"):
                        st.markdown('<div class="guardrail-warning">🛡️ <b>Guardrail Triggered:</b> This query was outside the financial domain scope and was blocked.</div>', unsafe_allow_html=True)

        # Quick question suggestions
        st.markdown('<div class="section-header">Suggested Questions</div>', unsafe_allow_html=True)
        suggestions = [
            "What were the total revenues last year?",
            "Summarize the key risk factors",
            "What is the debt-to-equity ratio?",
            "What are the main business segments?",
        ]
        cols = st.columns(4)
        for i, sug in enumerate(suggestions):
            with cols[i]:
                if st.button(sug, key=f"sug_{i}"):
                    st.session_state["pending_question"] = sug

        # Chat input
        st.markdown("<br>", unsafe_allow_html=True)
        with st.form("chat_form", clear_on_submit=True):
            col_input, col_send = st.columns([5, 1])
            with col_input:
                user_input = st.text_input(
                    "Ask a question about the document",
                    value=st.session_state.pop("pending_question", ""),
                    placeholder="e.g. What were the operating expenses in Q4?",
                    label_visibility="collapsed"
                )
            with col_send:
                submitted = st.form_submit_button("Send →")

        if submitted and user_input.strip():
            question = user_input.strip()
            st.session_state.chat_history.append({"role": "user", "content": question})

            # GUARDRAIL CHECK
            if guardrails_on and not is_financial_query(question):
                answer = "⛔ I'm configured to answer only financial and business-related questions about the uploaded document. Please ask about revenues, expenses, risks, strategy, or other financial topics."
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "guardrail_blocked": True,
                    "sources": []
                })
            else:
                with st.spinner("🔍 Retrieving relevant chunks and generating answer..."):
                    result = ask_question(
                        st.session_state.qa_chain,
                        question,
                        memory=st.session_state.memory
                    )
                    answer = result["answer"]
                    if disclaimer_on:
                        answer = inject_disclaimer(answer)
                    source_pages = result.get("source_pages", [])
                    add_to_memory(st.session_state.memory, question, answer)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": source_pages,
                        "guardrail_blocked": False
                    })

            st.rerun()

        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.session_state.memory = init_memory()
                st.rerun()

# ── TAB 3: ARCHITECTURE ────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">System Architecture</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#111827;border-radius:12px;padding:1.5rem;border:1px solid #1E3A5F;">
    <pre style="color:#B0BEC5;font-family:'IBM Plex Mono',monospace;font-size:0.78rem;line-height:1.8;margin:0;">
    ┌─────────────────────────────────────────────────────────────────┐
    │                      FinSight AI Architecture                   │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  USER                                                           │
    │   │                                                             │
    │   ▼                                                             │
    │  ┌──────────────┐    ┌─────────────────────────────────────┐   │
    │  │  Upload PDF  │───▶│         INGESTOR MODULE             │   │
    │  └──────────────┘    │  PyPDF → Text Extraction            │   │
    │                      │  RecursiveCharacterTextSplitter      │   │
    │                      │  → chunks (size=500, overlap=50)     │   │
    │                      └────────────┬────────────────────────┘   │
    │                                   │                             │
    │                                   ▼                             │
    │                      ┌─────────────────────────────────────┐   │
    │                      │     EMBEDDING + VECTOR STORE        │   │
    │                      │  OpenAIEmbeddings (ada-002)         │   │
    │                      │  FAISS Index (cosine similarity)     │   │
    │                      └────────────┬────────────────────────┘   │
    │                                   │                             │
    │          ┌────────────────────────┼────────────────────┐       │
    │          ▼                                             ▼        │
    │  ┌──────────────────┐                   ┌───────────────────┐  │
    │  │  SUMMARIZER      │                   │   Q&A CHAIN       │  │
    │  │  Map-Reduce      │                   │   ConversationalR │  │
    │  │  LLMChain        │                   │   etrievalChain   │  │
    │  └──────────────────┘                   └────────┬──────────┘  │
    │                                                   │             │
    │                                      ┌────────────▼──────────┐ │
    │                                      │     GUARDRAILS        │ │
    │                                      │  Domain classifier    │ │
    │                                      │  Disclaimer injector  │ │
    │                                      └────────────┬──────────┘ │
    │                                                   │             │
    │                                      ┌────────────▼──────────┐ │
    │                                      │    MEMORY MODULE      │ │
    │                                      │  ConversationBuffer   │ │
    │                                      │  WindowMemory (k=5)   │ │
    │                                      └───────────────────────┘ │
    └─────────────────────────────────────────────────────────────────┘
    </pre>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Tech Stack</div>', unsafe_allow_html=True)
        stack = {
            "Frontend": "Streamlit",
            "LLM": "GPT-4 (OpenAI)",
            "Embedding": "text-embedding-ada-002",
            "Vector Store": "FAISS (in-memory)",
            "Framework": "LangChain",
            "PDF Parsing": "PyPDF",
            "Summarization": "Map-Reduce LLMChain",
            "Memory": "ConversationBufferWindowMemory",
            "Guardrails": "Custom keyword + LLM classifier",
        }
        for tech, detail in stack.items():
            st.markdown(f"""<div class="pipeline-step">
                <div class="step-num">→</div>
                <b style="color:#64B5F6;min-width:130px">{tech}</b>
                <span style="color:#78909C">{detail}</span>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">Key LangChain Components</div>', unsafe_allow_html=True)
        components = [
            ("PyPDFLoader", "Load and parse PDF documents"),
            ("RecursiveCharacterTextSplitter", "Smart semantic chunking"),
            ("OpenAIEmbeddings", "Convert text chunks to vectors"),
            ("FAISS.from_documents", "Build in-memory vector index"),
            ("load_summarize_chain", "Map-Reduce summarization"),
            ("ConversationalRetrievalChain", "RAG Q&A with memory"),
            ("ConversationBufferWindowMemory", "Rolling conversation context"),
            ("PromptTemplate", "Custom system + user prompts"),
        ]
        for comp, desc in components:
            st.markdown(f"""<div class="pipeline-step">
                <div class="step-num">◆</div>
                <div>
                    <div style="font-family:'IBM Plex Mono',monospace;color:#00ACC1;font-size:0.78rem">{comp}</div>
                    <div style="color:#546E7A;font-size:0.75rem;margin-top:2px">{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)
