import json
import streamlit as st

from modules.agent import agent_answer
from modules.chatbot import answer_question, answer_question_stream
try:
    from modules.knowledge_base import load_knowledge_base
except (ImportError, KeyError):
    def load_knowledge_base(*_a, **_kw): return None  # type: ignore[misc]
from modules.llm_client import LLMConfigurationError
from modules.settings import get_text, init_session_settings
try:
    from modules.storage import save_chat_message, get_history, save_evaluation_event, clear_chat_history
except (ImportError, KeyError):
    def save_chat_message(*_a, **_kw): pass  # type: ignore[misc]
    def get_history(*_a, **_kw): return {"chat": []}  # type: ignore[misc]
    def save_evaluation_event(*_a, **_kw): pass  # type: ignore[misc]
    def clear_chat_history(): pass  # type: ignore[misc]
from modules.ui import inject_global_css

st.set_page_config(page_title="Chatbot", page_icon="💬", layout="wide")
inject_global_css()
init_session_settings()

st.markdown(f'<div class="page-header"><h2>{get_text("chatbot_title")}</h2></div>', unsafe_allow_html=True)
st.caption(get_text("chatbot_caption"))


@st.cache_resource
def get_kb():
    return load_knowledge_base()


kb = get_kb()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

    # Load chat history from SQLite on first load
    try:
        history_data = get_history(limit=20)
        if history_data.get("chat"):
            for msg in reversed(history_data["chat"]):
                sources = []
                if msg["sources"]:
                    try:
                        sources = json.loads(msg["sources"])
                    except json.JSONDecodeError:
                        sources = []
                st.session_state.chat_history.append({
                    "role": msg["role"],
                    "content": msg["content"],
                    "sources": sources,
                })
    except Exception as exc:
        st.warning(f"Could not load chat history: {exc}")

with st.sidebar:
    st.markdown("**Mode**")
    use_agent = st.radio(
        "Answer mode:",
        ["Agent (tool-calling)", "RAG only"],
        index=0,
        help=(
            "Agent mode calls real tools (device status, ping, alerts, anomaly scores) "
            "before generating an answer. RAG only uses the knowledge base."
        ),
    )

    st.divider()
    st.markdown(f"**{get_text('suggested_questions')}**")
    suggestions = [
        "What is the current status of all devices?",
        "Check if 8.8.8.8 is reachable and what latency it has.",
        "Are there any active alerts on the network right now?",
        "What causes high packet loss on a switch port?",
        "How do I troubleshoot DNS failures?",
        "Explain the difference between latency and bandwidth.",
    ]
    for q in suggestions:
        if st.button(q, key=f"suggest_{q[:25]}"):
            st.session_state.pending_query = q

    st.divider()
    st.markdown(
        '<p class="rtl-block">اسأل عن الشبكات أو الصق مخرجات ping</p>',
        unsafe_allow_html=True,
    )

    if use_agent == "RAG only":
        use_streaming = st.checkbox("Enable streaming responses", value=True)

    if st.button(get_text("clear_chat")):
        clear_chat_history()
        st.session_state.chat_history = []
        st.rerun()

# Render chat history
for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            st.caption(f"KB sources: {', '.join(turn['sources'])}")
        if turn.get("grounded") is False:
            st.caption("⚠ Answer not grounded in local knowledge base.")
        if turn.get("tool_trace"):
            with st.expander(f"🔧 Agent trace ({len(turn['tool_trace'])} tool calls)", expanded=False):
                for step in turn["tool_trace"]:
                    if "error" in step:
                        st.caption(f"Error: {step['error']}")
                    else:
                        st.markdown(
                            f'<div class="alert-feed ok">'
                            f'<span class="mono">Step {step["iteration"]} — {step["tool"]}</span>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        st.caption(f"Arguments: {json.dumps(step['arguments'])}")
                        st.text(step["result"][:400] + ("..." if len(step["result"]) > 400 else ""))

prompt = st.chat_input(get_text("ask_placeholder"))
if not prompt and st.session_state.get("pending_query"):
    prompt = st.session_state.pop("pending_query")

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if use_agent == "Agent (tool-calling)":
                    result = agent_answer(
                        prompt,
                        kb,
                        history=st.session_state.chat_history,
                        temperature=0.7,
                    )
                    answer = result["answer"]
                    sources = result["sources"]
                    tool_trace = result["tool_trace"]
                    grounded = bool(sources)

                    st.markdown(answer)

                    if sources:
                        st.caption(f"KB sources: {', '.join(sources)}")
                    if not grounded and not tool_trace:
                        st.caption("⚠ Answer not grounded in local knowledge base.")

                    if tool_trace:
                        with st.expander(f"🔧 Agent trace ({len(tool_trace)} tool calls)", expanded=False):
                            for step in tool_trace:
                                if "error" in step:
                                    st.caption(f"Error: {step['error']}")
                                else:
                                    st.markdown(
                                        f'<div class="alert-feed ok">'
                                        f'<span class="mono">Step {step["iteration"]} — {step["tool"]}</span>'
                                        f"</div>",
                                        unsafe_allow_html=True,
                                    )
                                    st.caption(f"Arguments: {json.dumps(step['arguments'])}")
                                    st.text(step["result"][:400] + ("..." if len(step["result"]) > 400 else ""))

                    assistant_turn = {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "grounded": grounded,
                        "tool_trace": tool_trace,
                    }
                    st.session_state.chat_history.append(assistant_turn)
                    save_chat_message("user", prompt)
                    save_chat_message("assistant", answer, sources)

                    # Log evaluation event
                    try:
                        save_evaluation_event(
                            event_type="agent_query",
                            query=prompt,
                            retrieval_hit=grounded,
                            retrieval_score=0.0,
                            latency_ms=result.get("latency_ms", 0),
                            metadata={
                                "tool_count": len(tool_trace),
                                "tools_used": [t.get("tool") for t in tool_trace if "tool" in t],
                            },
                        )
                    except Exception:
                        pass

                else:
                    # RAG-only mode
                    if use_streaming:
                        for answer, sources, grounded in answer_question_stream(
                            prompt,
                            kb,
                            history=st.session_state.chat_history,
                        ):
                            st.markdown(answer)
                            if sources:
                                st.caption(f"KB sources: {', '.join(sources)}")
                            if not grounded:
                                st.caption("⚠ Answer not grounded in local knowledge base.")

                            assistant_turn = {
                                "role": "assistant",
                                "content": answer,
                                "sources": sources,
                                "grounded": grounded,
                            }
                            st.session_state.chat_history.append(assistant_turn)
                            save_chat_message("user", prompt)
                            save_chat_message("assistant", answer, sources)
                            break
                    else:
                        result = answer_question(
                            prompt,
                            kb,
                            history=st.session_state.chat_history,
                        )
                        st.markdown(result["answer"])
                        if result["sources"]:
                            st.caption(f"KB sources: {', '.join(result['sources'])}")
                        if not result["grounded"]:
                            st.caption("⚠ Answer not grounded in local knowledge base.")

                        assistant_turn = {
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["sources"],
                            "grounded": result["grounded"],
                        }
                        st.session_state.chat_history.append(assistant_turn)
                        save_chat_message("user", prompt)
                        save_chat_message("assistant", result["answer"], result["sources"])

            except LLMConfigurationError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Chatbot error: {exc}")
