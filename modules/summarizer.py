"""
modules/summarizer.py
=====================
Implements Map-Reduce summarization using LangChain's load_summarize_chain.

Map-Reduce Pattern:
  MAP:    Each chunk → individual summary prompt → chunk summary
  REDUCE: All chunk summaries → combine prompt → final summary

This handles documents of ANY length by avoiding context window overflow.

Learning Points:
  - load_summarize_chain with chain_type="map_reduce"
  - Custom PromptTemplates for MAP and REDUCE steps
  - Financial domain-specific prompt engineering
"""

from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS


# ── Prompt Templates ──────────────────────────────────────────────────────────

MAP_PROMPT_TEMPLATE = """You are a financial analyst. Analyze the following excerpt from a financial document and extract the key information.

Focus on:
- Financial figures (revenues, expenses, profits, margins)
- Key business metrics and KPIs
- Strategic decisions or changes
- Risk factors or challenges mentioned
- Forward-looking statements

Document excerpt:
{text}

Concise financial summary of this excerpt:"""

REDUCE_TEMPLATE_EXECUTIVE = """You are a senior financial analyst writing an executive summary.
You have received partial summaries of a financial document. Synthesize them into a cohesive {length} executive summary.

Write in professional, clear language. Include specific numbers where available.
Partial summaries:
{text}

Executive Summary:"""

REDUCE_TEMPLATE_BULLETS = """You are a senior financial analyst.
Synthesize the following partial summaries into a {length} bullet-point summary of key findings.

Use this format:
• [Category]: [Finding with specific numbers if available]

Partial summaries:
{text}

Key Financial Findings:"""

REDUCE_TEMPLATE_RISKS = """You are a risk analyst reviewing a financial document.
Synthesize the following partial summaries focusing ONLY on risk factors, challenges, and uncertainties.

Format as numbered risk items with brief explanations.
Partial summaries:
{text}

Risk Factor Summary:"""

REDUCE_TEMPLATE_METRICS = """You are a financial data analyst.
Extract and synthesize all quantitative financial metrics from the following partial summaries.

Include: Revenue, Net Income, EBITDA, Margins, Cash Flow, Debt, Growth rates, etc.
Present in a structured format.

Partial summaries:
{text}

Financial Metrics Summary:"""

REDUCE_TEMPLATE_EARNINGS = """You are a financial analyst specializing in earnings analysis.
Synthesize the following into an earnings highlights report covering:
- Revenue and profit performance vs expectations
- Key growth drivers
- Management commentary highlights
- Guidance and outlook

Partial summaries:
{text}

Earnings Highlights:"""


def _get_length_word(length_str: str) -> str:
    mapping = {
        "Concise (150 words)": "concise ~150-word",
        "Standard (300 words)": "detailed ~300-word",
        "Detailed (500 words)": "comprehensive ~500-word"
    }
    return mapping.get(length_str, "detailed")


def _get_reduce_template(summary_type: str) -> str:
    if "Bullet" in summary_type:
        return REDUCE_TEMPLATE_BULLETS
    elif "Risk" in summary_type:
        return REDUCE_TEMPLATE_RISKS
    elif "Metrics" in summary_type:
        return REDUCE_TEMPLATE_METRICS
    elif "Earnings" in summary_type:
        return REDUCE_TEMPLATE_EARNINGS
    else:  # Executive Summary
        return REDUCE_TEMPLATE_EXECUTIVE


def generate_summary(
    vectorstore: FAISS,
    summary_type: str = "Executive Summary (2-3 paragraphs)",
    length: str = "Standard (300 words)"
) -> str:
    """
    Generate a document summary using Map-Reduce chain.

    Args:
        vectorstore: Indexed FAISS vectorstore with document chunks
        summary_type: Type of summary to generate
        length: Target summary length

    Returns:
        Generated summary string
    """
    # ── Retrieve all documents from vectorstore ───────────────────────────
    # For summarization we want ALL chunks, not just query-relevant ones
    all_docs = list(vectorstore.docstore._dict.values())

    # Limit to top 20 most content-rich chunks to manage token budget
    all_docs = sorted(all_docs, key=lambda d: len(d.page_content), reverse=True)[:20]

    # ── Build LLM ─────────────────────────────────────────────────────────
    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0.3,    # Low temp for factual summaries
        max_tokens=800
    )

    # ── Build Prompts ─────────────────────────────────────────────────────
    map_prompt = PromptTemplate(
        template=MAP_PROMPT_TEMPLATE,
        input_variables=["text"]
    )

    length_word = _get_length_word(length)
    reduce_template = _get_reduce_template(summary_type)
    reduce_prompt = PromptTemplate(
        template=reduce_template,
        input_variables=["text", "length"] if "{length}" in reduce_template else ["text"]
    )

    # ── Build and Run Map-Reduce Chain ────────────────────────────────────
    chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=reduce_prompt,
        verbose=False
    )

    result = chain.invoke({
        "input_documents": all_docs,
        "length": length_word
    })

    return result.get("output_text", "Unable to generate summary.")
