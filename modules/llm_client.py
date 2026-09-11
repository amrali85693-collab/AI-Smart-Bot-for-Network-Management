"""Thin wrapper around hosted LLM APIs (Groq primary, Gemini fallback) with clean type safety and injection points."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import streamlit as st

from modules.settings import get_llm_provider

GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"
GEMINI_DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 1


class LLMConfigurationError(Exception):
    """Raised when API keys or provider config are missing."""
    pass


class LLMExecutionError(Exception):
    """Raised when an LLM call fails after retries."""
    pass


def _get_secret(key: str) -> Optional[str]:
    """Safe retrieval of st.secrets."""
    try:
        return st.secrets.get(key)
    except (FileNotFoundError, KeyError, AttributeError):
        return None


def _call_groq(
    messages: List[Dict[str, str]],
    temperature: float,
    stream: bool = False,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Internal helper to invoke Groq API."""
    key = api_key or _get_secret("GROQ_API_KEY")
    if not key:
        raise LLMConfigurationError(
            "GROQ_API_KEY is not set. Add it to .streamlit/secrets.toml "
            "or Streamlit Cloud Secrets."
        )

    from groq import Groq

    client = Groq(api_key=key)
    selected_model = model or GROQ_DEFAULT_MODEL

    if stream:
        completion = client.chat.completions.create(
            model=selected_model,
            messages=messages,  # type: ignore
            temperature=temperature,
            max_tokens=1024,
            top_p=1,
            timeout=DEFAULT_TIMEOUT,
            stream=True,
        )
        full_content = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content
        return full_content or ""
    else:
        completion = client.chat.completions.create(
            model=selected_model,
            messages=messages,  # type: ignore
            temperature=temperature,
            max_tokens=1024,
            top_p=1,
            timeout=DEFAULT_TIMEOUT,
        )
        content = completion.choices[0].message.content
        if not content:
            reasoning = getattr(completion.choices[0].message, "reasoning", None)
            if reasoning:
                content = reasoning
        return content or ""


def _call_gemini(
    messages: List[Dict[str, str]],
    temperature: float,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Internal helper to invoke Google Gemini API."""
    key = api_key or _get_secret("GEMINI_API_KEY")
    if not key:
        raise LLMConfigurationError(
            "GEMINI_API_KEY is not set. Add it to .streamlit/secrets.toml "
            "or Streamlit Cloud Secrets."
        )

    import google.generativeai as genai

    genai.configure(api_key=key)
    selected_model = model or GEMINI_DEFAULT_MODEL
    gemini_model = genai.GenerativeModel(selected_model)

    system_parts: List[str] = []
    conversation: List[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_parts.append(content)
        elif role == "assistant":
            conversation.append(f"Assistant: {content}")
        else:
            conversation.append(f"User: {content}")

    prompt = ""
    if system_parts:
        prompt = "System instructions:\n" + "\n".join(system_parts) + "\n\n"
    prompt += "\n".join(conversation)

    response = gemini_model.generate_content(
        prompt,
        generation_config={"temperature": temperature, "max_output_tokens": 1024},
    )
    return response.text or ""


def get_llm_response(
    messages: List[Dict[str, str]],
    provider: Optional[str] = None,
    temperature: float = 0.7,
    stream: bool = False,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """
    Return assistant text response for a standard list of chat messages.
    Supports provider fallbacks automatically.
    """
    if provider is None:
        provider = get_llm_provider()

    provider_lower = provider.lower()
    providers = [provider_lower]
    if provider_lower == "groq":
        providers.append("gemini")
    elif provider_lower == "gemini":
        providers.append("groq")

    last_error: Optional[Exception] = None
    for current in providers:
        for attempt in range(MAX_RETRIES + 1):
            try:
                if current == "groq":
                    # Only Groq supports streaming currently
                    return _call_groq(
                        messages=messages,
                        temperature=temperature,
                        stream=stream,
                        api_key=api_key,
                        model=model,
                    )
                else:
                    return _call_gemini(
                        messages=messages,
                        temperature=temperature,
                        api_key=api_key,
                        model=model,
                    )
            except LLMConfigurationError as exc:
                last_error = exc
                # Try next provider if configuration error occurs
                break
            except Exception as exc:
                last_error = exc
                if attempt < MAX_RETRIES:
                    time.sleep(1.0)

    if isinstance(last_error, LLMConfigurationError):
        raise last_error
    raise LLMExecutionError(f"LLM request failed after retries: {last_error}") from last_error
