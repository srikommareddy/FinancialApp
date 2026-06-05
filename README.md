# 📊 FinSight AI
### Intelligent Financial Document Summarizer & Q&A Bot

> Capstone project for the **NVIDIA Gen AI LLM Associate Certification**  
> Demonstrates: RAG Pipeline · Map-Reduce Summarization · Guardrails · Conversation Memory

---

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

## ✨ Features

| Feature | Implementation |
|---|---|
| 📄 **PDF Ingestion** | PyPDF → RecursiveCharacterTextSplitter → FAISS |
| 🧠 **Document Summarization** | LangChain Map-Reduce chain (handles any length) |
| 💬 **Q&A Bot** | ConversationalRetrievalChain with source citations |
| 🛡️ **Guardrails** | Domain filter + auto disclaimer injection |
| 🔁 **Memory** | ConversationBufferWindowMemory (last 5 turns) |

---

## 🛠️ Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/finsight-ai.git
cd finsight-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml and add your OpenAI API key

# 4. Run
streamlit run app.py
```

---

## ☁️ Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo → branch `main` → file `app.py`
4. Under **Settings → Secrets**, add:
   ```toml
   OPENAI_API_KEY = "sk-your-key-here"
   ```
5. Click **Deploy** — done in ~2 minutes!

---

## 📁 Project Structure

```
finsight/
├── app.py                    ← Streamlit UI
├── requirements.txt          ← Pinned dependencies
├── .streamlit/
│   ├── config.toml           ← Theme & server settings
│   └── secrets.toml          ← Local secrets (gitignored)
└── modules/
    ├── ingestor.py           ← RAG ingestion pipeline
    ├── summarizer.py         ← Map-Reduce summarization
    ├── qa_chain.py           ← ConversationalRetrievalChain
    ├── guardrails.py         ← Safety filters
    └── memory.py             ← Conversation memory
```

---

## 🧪 Test Documents

Try uploading any of these public financial PDFs:
- [Apple 10-K](https://investor.apple.com/sec-filings/annual-reports)
- [Tesla Annual Report](https://ir.tesla.com)
- Any company earnings release PDF

---

*Built for educational purposes · NVIDIA Gen AI LLM Associate Certification*
