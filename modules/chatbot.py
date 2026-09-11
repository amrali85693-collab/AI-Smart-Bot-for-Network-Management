"""RAG orchestration: retrieve context, build prompt, call LLM."""

from __future__ import annotations

import time

try:
    from modules.knowledge_base import KnowledgeBase
except (ImportError, KeyError):
    KnowledgeBase = None  # type: ignore[misc,assignment]
from modules.llm_client import get_llm_response
try:
    from modules.storage import save_evaluation_event
except (ImportError, KeyError):
    def save_evaluation_event(*_a, **_kw): pass  # type: ignore[misc]

SYSTEM_PROMPT = (
    "You are a network operations assistant for administrators. "
    "Answer clearly and practically. When context from the knowledge base "
    "is provided, prefer it. If the user pastes diagnostics (ping, logs, "
    "config), interpret them step by step. "
    "IMPORTANT: You are ONLY allowed to answer questions related to: "
    "network management, network monitoring, network diagnostics, network security, "
    "network troubleshooting, network configuration, network protocols, "
    "network devices (routers, switches, firewalls), network performance, "
    "network alerts, and network infrastructure. "
    "If a question is completely outside this scope (e.g., cooking, sports, "
    "politics, entertainment, general knowledge not related to networking), "
    "politely refuse and state that you can only help with network-related topics."
)


def answer_question_stream(
    query: str,
    kb: KnowledgeBase,
    history: list[dict] | None = None,
    temperature: float = 0.7,
):
    """
    Generator function for streaming responses.
    Yields chunks of the answer as they arrive.
    """
    start_time = time.time()
    
    retrieved = kb.search(query, top_k=3)
    grounded = bool(retrieved)
    
    # Get retrieval score for evaluation
    retrieval_score = retrieved[0].get("score", 0.0) if retrieved else 0.0

    context_block = ""
    sources: list[str] = []
    if retrieved:
        snippets = []
        for entry in retrieved:
            topic = entry.get("topic", "Unknown")
            sources.append(topic)
            snippets.append(f"[{topic}] {entry.get('answer', '')}")
        context_block = "Knowledge base context:\n" + "\n\n".join(snippets)
    else:
        context_block = (
            "No closely matching knowledge base entry was found. "
            "Answer from general networking knowledge and say so briefly."
        )

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for turn in history[-6:]:
            messages.append({"role": turn["role"], "content": turn["content"]})

    user_content = f"{context_block}\n\nUser question:\n{query}"
    messages.append({"role": "user", "content": user_content})

    # Stream the response
    try:
        answer = get_llm_response(messages, temperature=temperature, stream=True)
    except Exception:
        # Fallback to non-streaming if streaming fails
        answer = get_llm_response(messages, temperature=temperature, stream=False)
    
    # Log evaluation metrics
    latency_ms = (time.time() - start_time) * 1000
    try:
        save_evaluation_event(
            event_type="chatbot_query",
            query=query,
            retrieval_score=retrieval_score,
            retrieval_hit=grounded,
            latency_ms=latency_ms,
            metadata={"sources": sources, "temperature": temperature, "streamed": True},
        )
    except Exception:
        # Silently fail if evaluation logging fails
        pass
    
    yield answer, sources, grounded


def answer_question(
    query: str,
    kb: KnowledgeBase,
    history: list[dict] | None = None,
    temperature: float = 0.7,
) -> dict:
    start_time = time.time()
    
    retrieved = kb.search(query, top_k=3)
    grounded = bool(retrieved)
    
    # Get retrieval score for evaluation
    retrieval_score = retrieved[0].get("score", 0.0) if retrieved else 0.0

    context_block = ""
    sources: list[str] = []
    if retrieved:
        snippets = []
        for entry in retrieved:
            topic = entry.get("topic", "Unknown")
            sources.append(topic)
            snippets.append(f"[{topic}] {entry.get('answer', '')}")
        context_block = "Knowledge base context:\n" + "\n\n".join(snippets)
    else:
        context_block = (
            "No closely matching knowledge base entry was found. "
            "Answer from general networking knowledge and say so briefly."
        )

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for turn in history[-6:]:
            messages.append({"role": turn["role"], "content": turn["content"]})

    user_content = f"{context_block}\n\nUser question:\n{query}"
    messages.append({"role": "user", "content": user_content})

    answer = get_llm_response(messages, temperature=temperature)
    
    # Log evaluation metrics
    latency_ms = (time.time() - start_time) * 1000
    try:
        save_evaluation_event(
            event_type="chatbot_query",
            query=query,
            retrieval_score=retrieval_score,
            retrieval_hit=grounded,
            latency_ms=latency_ms,
            metadata={"sources": sources, "temperature": temperature},
        )
    except Exception:
        # Silently fail if evaluation logging fails
        pass
    
    return {
        "answer": answer,
        "sources": sources,
        "grounded": grounded,
    }
