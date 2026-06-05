# FinSight AI — Capstone Project Instructor Guide
## NVIDIA Gen AI LLM Associate Certification

---

## 🎯 Project Overview

**FinSight AI** is a production-grade financial document analysis app demonstrating:
- **RAG Pipeline** (Document Q&A with source citations)
- **Map-Reduce Summarization** (handles any document length)
- **Guardrails** (domain filtering + disclaimer injection)
- **Conversation Memory** (multi-turn context management)

**Tech Stack:** Python · LangChain · OpenAI GPT-4 · FAISS · Streamlit

---

## 📁 Project Structure

```
finsight/
├── app.py                  ← Main Streamlit UI
├── requirements.txt        ← Python dependencies
├── .env                    ← API keys (never commit!)
└── modules/
    ├── __init__.py
    ├── ingestor.py         ← PDF → Chunks → FAISS index
    ├── summarizer.py       ← Map-Reduce summarization chain
    ├── qa_chain.py         ← ConversationalRetrievalChain (RAG Q&A)
    ├── guardrails.py       ← Input/output safety filters
    └── memory.py           ← ConversationBufferWindowMemory
```

---

## 🛠️ Setup Instructions

### 1. Prerequisites
```bash
Python 3.9+
pip
OpenAI API Key (https://platform.openai.com)
```

### 2. Install Dependencies
```bash
cd finsight
pip install -r requirements.txt
```

### 3. Configure API Key
Create a `.env` file:
```
OPENAI_API_KEY=sk-your-key-here
```
Or enter it directly in the Streamlit sidebar when running the app.

### 4. Run the App
```bash
streamlit run app.py
```
The app opens at `http://localhost:8501`

### 5. Test Document
Download any public financial document to test:
- Apple 10-K: https://investor.apple.com/sec-filings/annual-reports
- Tesla Annual Report: https://ir.tesla.com
- Any company's PDF annual report or earnings release

---

## 🏗️ Architecture Deep-Dive

### Module 1: Ingestor (`modules/ingestor.py`)

**What it does:** Converts a raw PDF into a searchable vector index.

**Key LangChain Components:**
```python
PyPDFLoader(path)                    # Extracts text from each PDF page
RecursiveCharacterTextSplitter(      # Splits text into chunks
    chunk_size=500*4,                # ~500 tokens per chunk
    chunk_overlap=50*4,              # 50-token overlap
    separators=["\n\n", "\n", ". "]  # Tries paragraph → line → sentence
)
OpenAIEmbeddings()                   # text-embedding-ada-002 (1536-dim)
FAISS.from_documents(chunks, emb)    # Builds cosine similarity index
```

**Teaching Points:**
- Why chunk? LLMs have token limits; a 100-page PDF can't fit in one prompt
- Why overlap? Prevents losing context at chunk boundaries
- Why FAISS? Fast approximate nearest neighbor search (sub-millisecond)
- 1536 dimensions = very rich semantic representation

---

### Module 2: Summarizer (`modules/summarizer.py`)

**What it does:** Produces abstractive summaries using Map-Reduce.

**Map-Reduce Pattern:**
```
PDF chunks (20 most content-rich)
    │
    ▼  MAP STEP (parallel)
[chunk 1] → LLM → summary_1
[chunk 2] → LLM → summary_2
    ...
[chunk N] → LLM → summary_N
    │
    ▼  REDUCE STEP
[summary_1 + summary_2 + ... + summary_N] → LLM → FINAL SUMMARY
```

**Key LangChain API:**
```python
chain = load_summarize_chain(
    llm=llm,
    chain_type="map_reduce",   # Other options: "stuff", "refine"
    map_prompt=map_prompt,     # Applied to each chunk
    combine_prompt=reduce_prompt  # Applied to combined summaries
)
result = chain.invoke({"input_documents": all_docs, "length": "300-word"})
```

**Teaching Points:**
- `chain_type="stuff"`: Put all text in one prompt (only for short docs)
- `chain_type="map_reduce"`: Best for long docs, parallelizable
- `chain_type="refine"`: Best for narrative coherence, sequential
- Temperature = 0.3 for factual summaries (lower = more deterministic)

---

### Module 3: Q&A Chain (`modules/qa_chain.py`)

**What it does:** Answers multi-turn questions using RAG.

**ConversationalRetrievalChain Flow:**
```
User Question + Chat History
    │
    ▼  Step 1: Condense Question
    LLM rewrites question to be standalone
    (e.g., "What about that?" → "What is Apple's revenue growth rate?")
    │
    ▼  Step 2: Retrieve
    FAISS finds top-K most similar chunks
    │
    ▼  Step 3: Generate
    GPT-4 answers using retrieved context + system prompt
    │
    ▼  Return answer + source page numbers
```

**Key Code:**
```python
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    return_source_documents=True,  # enables citations
    combine_docs_chain_kwargs={"prompt": qa_prompt}
)
result = chain.invoke({"question": "What were revenues?"})
answer = result["answer"]
source_docs = result["source_documents"]  # list of retrieved chunks
```

**Teaching Points:**
- Without the "condense question" step, follow-up questions like "What about expenses?" would fail (no prior context)
- `return_source_documents=True` enables citations/grounding
- Temperature = 0.1 for Q&A (maximum factual accuracy)

---

### Module 4: Guardrails (`modules/guardrails.py`)

**What it does:** Two-layer safety system.

**Input Guardrail (Domain Filter):**
```
Query → Off-topic regex check → Financial keyword check → Allow/Block
```

**Output Guardrail (Disclaimer):**
```
LLM Answer → Append standard financial disclaimer → Serve to user
```

**Code Pattern:**
```python
# Before calling the LLM:
if guardrails_on and not is_financial_query(question):
    return "⛔ Please ask financial questions only."

# After getting the LLM answer:
if disclaimer_on:
    answer = inject_disclaimer(answer)
```

**Teaching Points:**
- Guardrails are a PRODUCTION REQUIREMENT, not optional
- Input guardrails save API costs (no LLM call for blocked queries)
- NVIDIA NeMo Guardrails (Colang) is the enterprise version of this pattern
- Disclaimer injection = output guardrail protecting against misuse

---

### Module 5: Memory (`modules/memory.py`)

**What it does:** Maintains rolling conversation context.

**Why Memory Matters:**
```
Turn 1: "What were Apple's revenues?"         → LLM answers correctly
Turn 2: "How does that compare to last year?" → Without memory: LLM asks "compare what?"
                                               → With memory: LLM knows "that" = revenues
```

**ConversationBufferWindowMemory (k=5):**
```python
memory = ConversationBufferWindowMemory(
    k=5,                    # Keep only last 5 turns
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)
```

**Teaching Points:**
- k=5 means max 10 messages (5 human + 5 AI) in context
- Full history would overflow context window for long sessions
- Alternative: ConversationSummaryMemory (compresses old turns)
- Alternative: VectorStoreRetrieverMemory (semantic retrieval of relevant past turns)

---

## 🧪 Classroom Exercise Plan

### Exercise 1: Basic RAG (15 min)
1. Upload an annual report PDF
2. Use the Summarizer → Executive Summary
3. Discuss: Which chunks were used? Why those?
4. Change chunk_size slider and re-summarize. What changes?

### Exercise 2: Q&A Exploration (20 min)
1. Ask 5 questions from the "Suggested Questions"
2. Note the source page badges — verify against the actual PDF
3. Ask a follow-up question that requires prior context ("What about that metric last year?")
4. Toggle guardrails OFF and ask an off-topic question. Then turn them ON.

### Exercise 3: Guardrail Testing (10 min)
Ask these queries with guardrails ON:
- "What's the weather today?" → should be BLOCKED
- "Tell me a joke" → should be BLOCKED
- "What are the operating expenses?" → should be ALLOWED
- "What is the risk exposure?" → should be ALLOWED
Discuss: How would you improve the guardrail logic?

### Exercise 4: Memory Demonstration (10 min)
1. Ask: "What were the total revenues?"
2. Ask: "How does that compare to the previous year?"
3. Ask: "What drove the change?"
4. Click "Clear Chat History" and ask question 2 again. What happens?

### Exercise 5: Modify the Code (20 min — advanced)
Ask students to:
1. Add a new summary type: "SWOT Analysis"
2. Change k=5 to k=3 in memory and test if it breaks multi-turn context
3. Add a new off-topic pattern to guardrails (e.g., sports)

---

## 📊 What Each Component Teaches (Exam Mapping)

| Module | Concept Demonstrated | NVIDIA Exam Topic |
|--------|---------------------|-------------------|
| ingestor.py | RAG ingestion pipeline | LLM Applications Integration |
| ingestor.py | Chunking strategies | Context Window Management |
| summarizer.py | Map-Reduce chain | Summarization patterns |
| summarizer.py | Prompt templates | Prompt Engineering |
| qa_chain.py | ConversationalRetrievalChain | Document Q&A Systems |
| qa_chain.py | Source citations | Faithfulness/Grounding |
| guardrails.py | Input/output guardrails | Safety & Reliability |
| memory.py | Window memory | Context Management |
| app.py | Full integration | End-to-end LLM app |

---

## 🔧 Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| `AuthenticationError` | Invalid API key | Check .env file or sidebar input |
| `RateLimitError` | Too many requests | Add `time.sleep(1)` between chunks |
| Empty summary | No text extracted | Check if PDF is image-based (needs OCR) |
| Slow ingestion | Large PDF | Reduce chunk count or use async |
| Memory not working | Wrong output_key | Ensure `output_key="answer"` in memory |

---

## 🚀 Extension Ideas (for advanced students)

1. **Add Reranking**: Use a cross-encoder to rerank retrieved chunks before generation
2. **Multi-Document RAG**: Allow uploading multiple PDFs; compare companies
3. **Agent Tool**: Add a live stock price tool using `yfinance`
4. **Export Chat**: Add a "Download Chat History" button
5. **Evaluation**: Implement ROUGE scoring for summary quality
6. **Streaming**: Use `streaming=True` in ChatOpenAI for real-time output
7. **ChromaDB**: Replace FAISS with ChromaDB for persistent storage

---

*FinSight AI Capstone Project · NVIDIA Gen AI LLM Associate Certification*
