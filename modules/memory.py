"""
modules/memory.py
=================
Conversation memory management for FinSight AI Q&A Bot.

Uses LangChain's ConversationBufferWindowMemory to maintain a rolling
window of the last K conversation turns, preventing context window overflow.

Learning Points:
  - ConversationBufferWindowMemory: keeps last k=5 turns
  - Why memory matters: LLMs are stateless; memory injects prior context
  - Window vs Full history vs Summary memory trade-offs
"""

from langchain.memory import ConversationBufferWindowMemory


def init_memory(k: int = 5) -> ConversationBufferWindowMemory:
    """
    Initialize a new conversation memory buffer.

    ConversationBufferWindowMemory keeps only the last k interactions,
    preventing the context window from overflowing on long conversations.

    Args:
        k: Number of recent conversation turns to retain (default: 5)

    Returns:
        Initialized ConversationBufferWindowMemory
    """
    return ConversationBufferWindowMemory(
        k=k,
        memory_key="chat_history",          # Key expected by ConversationalRetrievalChain
        return_messages=True,               # Return as Message objects (not string)
        output_key="answer"                 # Which chain output to save as AI response
    )


def add_to_memory(memory: ConversationBufferWindowMemory,
                  user_message: str,
                  ai_message: str) -> None:
    """
    Manually save a user/AI exchange to memory.

    Used when we handle guardrail blocks outside the chain —
    we still want to record the interaction for context.

    Args:
        memory: Active memory instance
        user_message: User's question
        ai_message: AI's response (or guardrail message)
    """
    memory.save_context(
        {"input": user_message},
        {"answer": ai_message}
    )


def get_history(memory: ConversationBufferWindowMemory) -> list:
    """
    Retrieve formatted conversation history from memory.

    Returns:
        List of message dicts with role and content
    """
    messages = memory.chat_memory.messages
    history = []
    for msg in messages:
        role = "user" if msg.type == "human" else "assistant"
        history.append({"role": role, "content": msg.content})
    return history
