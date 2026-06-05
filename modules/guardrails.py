"""
modules/guardrails.py
=====================
Input and output guardrails for FinSight AI.

Implements two guardrail layers:
  1. INPUT GUARDRAIL: Domain classifier — blocks non-financial queries
  2. OUTPUT GUARDRAIL: Disclaimer injector — appends regulatory disclaimer

Learning Points:
  - Guardrails as a production safety mechanism for LLM apps
  - Keyword-based fast-path classification (avoids extra LLM call)
  - NVIDIA NeMo Guardrails concept demonstration (simplified)
"""

import re


# ── Financial Domain Keywords ──────────────────────────────────────────────
FINANCIAL_KEYWORDS = {
    # Core financial terms
    "revenue", "profit", "loss", "income", "expense", "cost", "margin",
    "ebitda", "ebit", "earnings", "dividend", "equity", "debt", "liability",
    "asset", "cash", "flow", "balance", "sheet", "quarter", "annual",
    "fiscal", "budget", "forecast", "guidance", "outlook",
    # Business metrics
    "sales", "growth", "market", "share", "stock", "valuation", "price",
    "acquisition", "merger", "subsidiary", "segment", "portfolio",
    # Risk and compliance
    "risk", "regulatory", "compliance", "audit", "litigation", "exposure",
    # Document types
    "10-k", "10k", "annual report", "earnings", "financial statement",
    "balance sheet", "income statement", "cash flow statement",
    # Business general
    "strategy", "operations", "business", "company", "corporation",
    "employee", "headcount", "customer", "product", "service",
    "investment", "capital", "return", "performance",
}

# Patterns that are clearly off-topic for a financial assistant
OFF_TOPIC_PATTERNS = [
    r"\b(recipe|cook|food|meal)\b",
    r"\b(movie|film|actor|actress|celebrity)\b",
    r"\b(weather|temperature|forecast)\b",
    r"\b(joke|funny|humor|meme)\b",
    r"\b(sports|football|basketball|cricket)\b",
    r"\b(poem|poetry|song|music|lyrics)\b",
    r"\b(travel|vacation|hotel|flight)\b",
    r"\b(medical|doctor|prescription|health symptom)\b",
]


def is_financial_query(query: str) -> bool:
    """
    Input Guardrail: Classify whether a query is financial/business in nature.

    Uses a two-layer approach:
      1. Fast path: Check for off-topic patterns (regex)
      2. Main path: Check for presence of financial keywords

    Args:
        query: User's input query string

    Returns:
        True if the query is financial/business in scope, False otherwise
    """
    query_lower = query.lower()

    # Layer 1: Fast reject — clearly off-topic patterns
    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, query_lower):
            return False

    # Layer 2: Allow questions about the document generically
    # (e.g., "What does this document say about..." is always in scope)
    generic_document_phrases = [
        "document", "report", "file", "pdf", "text", "section",
        "what does", "tell me about", "explain", "summarize", "describe"
    ]
    for phrase in generic_document_phrases:
        if phrase in query_lower:
            return True

    # Layer 3: Check for financial keyword presence
    words = re.findall(r'\b\w+\b', query_lower)
    for word in words:
        if word in FINANCIAL_KEYWORDS:
            return True

    # Layer 4: If query is short and generic (e.g., "What is it?"), allow it
    if len(words) <= 6:
        return True

    # Default: block if no financial signal detected
    return False


def inject_disclaimer(answer: str) -> str:
    """
    Output Guardrail: Append a standard financial disclaimer to LLM responses.

    This simulates a real-world compliance requirement where AI-generated
    financial information must carry an advisory disclaimer.

    Args:
        answer: LLM-generated answer string

    Returns:
        Answer with disclaimer appended
    """
    disclaimer = (
        "\n\n---\n*⚠️ Financial Disclaimer: This analysis is AI-generated "
        "from the uploaded document for informational purposes only. "
        "It does not constitute investment or financial advice. "
        "Always consult a qualified financial professional before making decisions.*"
    )
    return answer + disclaimer
