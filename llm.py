"""
llm.py
------
Calls the chat-completion API for two tasks:
  1. answer_question  — grounded Q&A that only uses retrieved passages.
  2. compare_paper     — structured field extraction for the comparison table.

Both functions are strict about not inventing information: the prompts
explicitly instruct the model to say so when the provided text doesn't
contain an answer, and the comparison extractor must write "Not specified"
rather than guess.
"""

from __future__ import annotations

import json
import os
from typing import List

from openai import OpenAI, APIError, APIConnectionError, AuthenticationError

from pdf_processor import Chunk

CHAT_MODEL = "gpt-4o-mini"

NO_ANSWER_PHRASE = (
    "The retrieved passages do not contain enough information to answer this question."
)

COMPARISON_FIELDS = [
    "research_question",
    "methods",
    "datasets",
    "evaluation_metrics",
    "limitations",
]


class LLMError(Exception):
    """Raised when the chat-completion API call fails for a reason the UI should surface."""


def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise LLMError(
            "OPENAI_API_KEY is not set. Add it to your .env file (see .env.example)."
        )
    return OpenAI(api_key=api_key)


def _chat(messages: List[dict], temperature: float = 0.0, json_mode: bool = False) -> str:
    client = _get_client()
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )
    except AuthenticationError as e:
        raise LLMError("Authentication with the LLM API failed — check OPENAI_API_KEY.") from e
    except APIConnectionError as e:
        raise LLMError("Could not connect to the LLM API. Check your network connection.") from e
    except APIError as e:
        raise LLMError(f"LLM API returned an error: {e}") from e
    except Exception as e:
        raise LLMError(f"Unexpected error while calling the LLM: {e}") from e

    return resp.choices[0].message.content or ""


def _format_passages(chunks_with_scores: List[tuple[Chunk, float]]) -> str:
    blocks = []
    for i, (chunk, _score) in enumerate(chunks_with_scores, start=1):
        blocks.append(
            f"[Passage {i} | Paper: {chunk.paper_title} | Page: {chunk.page_number}]\n{chunk.text}"
        )
    return "\n\n".join(blocks)


def answer_question(question: str, chunks_with_scores: List[tuple[Chunk, float]]) -> str:
    """
    Answer `question` using ONLY the given retrieved (chunk, score) passages.
    The model is instructed to cite [Paper, Page] after each claim and to say
    so plainly if the passages don't support an answer.
    """
    if not chunks_with_scores:
        return NO_ANSWER_PHRASE

    passages = _format_passages(chunks_with_scores)

    system_prompt = (
        "You are a research assistant that answers questions using ONLY the "
        "provided passages from academic papers. Rules:\n"
        "1. Do not use outside knowledge or anything not contained in the passages.\n"
        "2. After every factual claim, cite it like this: (Paper Title, p. X).\n"
        "3. If the passages do not contain enough information to answer the "
        f"question, respond with exactly: \"{NO_ANSWER_PHRASE}\" — do not guess "
        "or partially answer.\n"
        "4. Be concise and precise. Do not fabricate citations, page numbers, "
        "or findings that are not in the passages."
    )
    user_prompt = f"Passages:\n\n{passages}\n\nQuestion: {question}"

    return _chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    ).strip()


def compare_paper(paper_title: str, chunks_with_scores: List[tuple[Chunk, float]]) -> dict:
    """
    Extract structured comparison fields for one paper from its most relevant
    passages. Any field not supported by the passages is set to "Not specified".
    Returns a dict with keys: research_question, methods, datasets,
    evaluation_metrics, limitations (plus "paper_title").
    """
    fallback = {"paper_title": paper_title, **{f: "Not specified" for f in COMPARISON_FIELDS}}

    if not chunks_with_scores:
        return fallback

    passages = _format_passages(chunks_with_scores)

    system_prompt = (
        "You extract structured metadata about a research paper from provided "
        "passages only. Return a JSON object with exactly these keys: "
        f"{', '.join(COMPARISON_FIELDS)}. For each key, write a short, factual "
        "summary (1-2 sentences) grounded strictly in the passages. If the "
        "passages do not mention that field, set its value to exactly "
        "\"Not specified\" — never guess or infer. Do not include any keys "
        "other than the ones listed."
    )
    user_prompt = (
        f"Paper title: {paper_title}\n\nPassages:\n\n{passages}\n\n"
        "Extract the fields now as a JSON object."
    )

    raw = _chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        json_mode=True,
    )

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return fallback

    result = {"paper_title": paper_title}
    for field in COMPARISON_FIELDS:
        value = data.get(field)
        if not value or not isinstance(value, str) or not value.strip():
            value = "Not specified"
        result[field] = value.strip()
    return result
