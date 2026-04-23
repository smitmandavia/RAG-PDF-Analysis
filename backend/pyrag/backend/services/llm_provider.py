"""
PyRAG — LLM Provider
Abstraction layer for LLM calls.
Supports Claude API and Ollama, switchable via config.
"""

import httpx
import json
from config import (
    LLM_PROVIDER, CLAUDE_API_KEY, CLAUDE_MODEL,
    OLLAMA_BASE_URL, OLLAMA_MODEL
)


import datetime

SYSTEM_PROMPT = f"""You are PyRAG, an intelligent document assistant. Your job is to answer questions accurately based ONLY on the provided context from uploaded documents.

Today's date is {datetime.date.today().strftime('%B %d, %Y')}. Use this to calculate durations when dates like "Present" or "current" appear in documents.

Rules:
1. Answer ONLY from the provided context. Never make up information.
2. If the context doesn't contain enough information to answer, say "I don't have enough information in the uploaded documents to answer this question."
3. Always cite your sources using the format [Doc: filename, Page: X].
4. Be concise but thorough. Use bullet points for lists.
5. If multiple documents contain relevant information, synthesize the answer and cite all sources.
6. Preserve technical accuracy — don't oversimplify unless asked.
7. Use recent conversation only to understand follow-up questions. Do not treat conversation history as evidence; factual claims must come from uploaded document context."""


def _clip(text: str, max_chars: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def _format_chat_history(chat_history: list[dict] | None) -> str:
    if not chat_history:
        return ""

    turns = []
    for turn in chat_history:
        turns.append(f"User: {_clip(turn.get('question', ''), 300)}")
        turns.append(f"Assistant: {_clip(turn.get('answer', ''), 500)}")
    return "\n".join(turns)


def build_prompt(question: str, context_chunks: list[dict], chat_history: list[dict] | None = None) -> str:
    """Build the prompt with retrieved context."""
    context_parts = []

    for i, chunk in enumerate(context_chunks, 1):
        meta = chunk['metadata']
        score = chunk['similarity']
        section = meta.get('section_title') or 'Overview'
        context_parts.append(
            f"[Source {i}: {meta['document_name']}, Page {meta['page_number']}, Section: {section}] "
            f"(Relevance: {score:.0%})\n{chunk['text']}"
        )

    context_text = "\n\n---\n\n".join(context_parts)
    history_text = _format_chat_history(chat_history)
    history_section = ""
    if history_text:
        history_section = f"""Recent conversation:
{history_text}

Use this only to resolve pronouns, references, and follow-up intent.

---

"""

    return f"""{history_section}Context from uploaded documents:

{context_text}

---

Question: {question}

Provide a clear, accurate answer based on the context above. Cite sources."""


async def generate_answer(question: str, context_chunks: list[dict], chat_history: list[dict] | None = None) -> str:
    """
    Generate an answer using the configured LLM.
    Routes to Claude or Ollama based on config.
    Falls back to raw context if no LLM is available.
    """
    prompt = build_prompt(question, context_chunks, chat_history=chat_history)

    if LLM_PROVIDER == "claude":
        if not CLAUDE_API_KEY:
            return _fallback_answer(question, context_chunks)
        return await _call_claude(prompt)
    elif LLM_PROVIDER == "ollama":
        try:
            return await _call_ollama(prompt)
        except Exception:
            return _fallback_answer(question, context_chunks)
    elif LLM_PROVIDER == "fallback":
        return _fallback_answer(question, context_chunks)
    else:
        return _fallback_answer(question, context_chunks)


def _fallback_answer(question: str, context_chunks: list[dict]) -> str:
    """Fallback: return retrieved context directly (no LLM needed)."""
    parts = [f"**Retrieved context for:** {question}\n"]
    parts.append("_(No LLM configured — showing raw retrieved chunks. Set ANTHROPIC_API_KEY or LLM_PROVIDER=ollama for AI-generated answers.)_\n")

    for i, chunk in enumerate(context_chunks, 1):
        meta = chunk['metadata']
        score = chunk['similarity']
        section = meta.get('section_title') or 'Overview'
        parts.append(f"**[Source {i}: {meta['document_name']}, Page {meta['page_number']}, Section: {section}]** (Relevance: {score:.0%})")
        parts.append(chunk['text'][:500])
        parts.append("")

    return "\n\n".join(parts)


async def _call_claude(prompt: str) -> str:
    """Call Claude API."""
    if not CLAUDE_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY not set. "
            "Set it via: export ANTHROPIC_API_KEY=your_key_here"
        )

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": CLAUDE_API_KEY,
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": CLAUDE_MODEL,
                "max_tokens": 2048,
                "system": SYSTEM_PROMPT,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
        )

        if response.status_code != 200:
            error_body = response.text
            raise Exception(f"Claude API error ({response.status_code}): {error_body}")

        data = response.json()
        return data["content"][0]["text"]


async def _call_ollama(prompt: str) -> str:
    """Call local Ollama instance."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 2048
                    }
                }
            )

            if response.status_code != 200:
                raise Exception(f"Ollama error ({response.status_code}): {response.text}")

            data = response.json()
            return data["response"]

        except httpx.ConnectError:
            raise Exception(
                f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
                "Make sure Ollama is running: ollama serve"
            )
