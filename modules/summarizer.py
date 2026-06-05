"""
modules/summarizer.py
=====================
Implements Map-Reduce summarization using LangChain's load_summarize_chain.
"""

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

# langchain-classic provides the legacy chains/prompts API
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_classic.prompts import PromptTemplate


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
Synthesize the partial summaries into a cohesive {length} executive summary.
Include specific numbers where available. Write in professional language.

Partial summaries:
{text}

Executive Summary:"""

REDUCE_TEMPLATE_BULLETS = """You are a senior financial analyst.
Synthesize the following partial summaries into a {length} bullet-point summary.

Format:
• [Category]: [Finding with specific numbers if available]

Partial summaries:
{text}

Key Financial Findings:"""

REDUCE_TEMPLATE_RISKS = """You are a risk analyst reviewing a financial document.
Synthesize the partial summaries focusing ONLY on risk factors and challenges.
Format as numbered risk items.

Partial summaries:
{text}

Risk Factor Summary:"""

REDUCE_TEMPLATE_METRICS = """You are a financial data analyst.
Extract and synthesize all quantitative financial metrics from the partial summaries.
Include: Revenue, Net Income, EBITDA, Margins, Cash Flow, Debt, Growth rates.

Partial summaries:
{text}

Financial Metrics Summary:"""

REDUCE_TEMPLATE_EARNINGS = """You are a financial analyst specializing in earnings analysis.
Synthesize the following into an earnings highlights report covering:
- Revenue and profit performance
- Key growth drivers
- Management commentary
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
    else:
        return REDUCE_TEMPLATE_EXECUTIVE


def generate_summary(vectorstore: FAISS, summary_type: str = "Executive Summary (2-3 paragraphs)", length: str = "Standard (300 words)") -> str:
    """Generate a document summary using Map-Reduce chain."""
    all_docs = list(vectorstore.docstore._dict.values())
    all_docs = sorted(all_docs, key=lambda d: len(d.page_content), reverse=True)[:20]

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3, max_tokens=800)

    map_prompt = PromptTemplate(template=MAP_PROMPT_TEMPLATE, input_variables=["text"])

    length_word = _get_length_word(length)
    reduce_template = _get_reduce_template(summary_type)
    has_length = "{length}" in reduce_template
    reduce_prompt = PromptTemplate(
        template=reduce_template,
        input_variables=["text", "length"] if has_length else ["text"]
    )

    chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=reduce_prompt,
        verbose=False
    )

    invoke_input = {"input_documents": all_docs}
    if has_length:
        invoke_input["length"] = length_word

    result = chain.invoke(invoke_input)
    return result.get("output_text", "Unable to generate summary.")
