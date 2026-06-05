"""
modules/memory.py
=================
Conversation memory management for FinSight AI Q&A Bot.
"""

# langchain-classic provides the legacy memory API
from langchain_classic.memory import ConversationBufferWindowMemory


def init_memory(k: int = 5) -> ConversationBufferWindowMemory:
    """Initialize a rolling window conversation memory (last k turns)."""
    return ConversationBufferWindowMemory(
        k=k,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )


def add_to_memory(memory: ConversationBufferWindowMemory, user_message: str, ai_message: str) -> None:
    """Manually save a user/AI exchange to memory."""
    memory.save_context(
        {"input": user_message},
        {"answer": ai_message}
    )


def get_history(memory: ConversationBufferWindowMemory) -> list:
    """Retrieve formatted conversation history from memory."""
    messages = memory.chat_memory.messages
    history = []
    for msg in messages:
        role = "user" if msg.type == "human" else "assistant"
        history.append({"role": role, "content": msg.content})
    return history
