"""Bridge dashboard alerts into LLM-generated troubleshooting explanations."""

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

DIAGNOSTICS_SYSTEM = (
    "You are a senior network engineer writing incident notes for operators. "
    "Be concise, factual, and actionable. Use plain language. "
    "IMPORTANT: You are ONLY allowed to answer questions related to: "
    "network management, network monitoring, network diagnostics, network security, "
    "network troubleshooting, network configuration, network protocols, "
    "network devices (routers, switches, firewalls), network performance, "
    "network alerts, and network infrastructure. "
    "If a question is completely outside this scope (e.g., cooking, sports, "
    "politics, entertainment, general knowledge not related to networking), "
    "politely refuse and state that you can only help with network-related topics."
)


def _alert_prompt(alert: dict, kb_context: str = "") -> str:
    level = alert.get("level", "unknown")
    device = alert.get("device", "unknown")
    metric = alert.get("metric", "unknown")
    value = alert.get("value", "n/a")
    message = alert.get("message", "")

    kb_block = ""
    if kb_context:
        kb_block = f"\n\nRelevant knowledge base notes:\n{kb_context}"

    return f"""An alert was raised on the network dashboard:

- Severity: {level}
- Device: {device}
- Metric: {metric}
- Value: {value}
- Alert message: {message}{kb_block}

Provide:
1. A likely cause in plain language (2-3 sentences).
2. Three to four concrete troubleshooting steps, numbered.
3. For each step, note whether it relies on standard networking practice or would benefit from local device documentation.

Keep the total response under 250 words."""


def explain_alert(alert: dict, kb: KnowledgeBase | None = None) -> str:
    start_time = time.time()
    
    kb_context = ""
    retrieval_score = 0.0
    retrieval_hit = False
    
    if kb is not None:
        query = f"{alert.get('metric', '')} {alert.get('message', '')}"
        hits = kb.search(query, top_k=2)
        if hits:
            kb_context = "\n".join(
                f"[{h.get('topic')}] {h.get('answer', '')}" for h in hits
            )
            retrieval_score = hits[0].get("score", 0.0)
            retrieval_hit = True

    messages = [
        {"role": "system", "content": DIAGNOSTICS_SYSTEM},
        {"role": "user", "content": _alert_prompt(alert, kb_context)},
    ]
    explanation = get_llm_response(messages, temperature=0.4)
    
    # Log evaluation metrics for alert-to-explanation latency
    latency_ms = (time.time() - start_time) * 1000
    try:
        save_evaluation_event(
            event_type="alert_explanation",
            query=alert.get("message", ""),
            retrieval_score=retrieval_score,
            retrieval_hit=retrieval_hit,
            latency_ms=latency_ms,
            metadata={"device": alert.get("device"), "level": alert.get("level")},
        )
    except Exception:
        # Silently fail if evaluation logging fails
        pass
    
    return explanation


def alert_to_chat_prompt(alert: dict) -> str:
    """Format an alert as a user message for the chatbot page."""
    return (
        f"I have this network alert:\n"
        f"- Severity: {alert.get('level')}\n"
        f"- Device: {alert.get('device')}\n"
        f"- Metric: {alert.get('metric')}\n"
        f"- Value: {alert.get('value')}\n"
        f"- Message: {alert.get('message')}\n\n"
        "What is the likely cause and what should I check first?"
    )
