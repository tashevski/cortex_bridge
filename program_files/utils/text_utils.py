#!/usr/bin/env python3
"""Light-weight text utilities shared across the program pipeline.

The functions collected here are completely independent of external
services – they operate purely on plain Python objects so that they can
be reused freely without introducing additional run-time dependencies.
"""

from typing import List, Dict, Any

# Common interrogative indicators
_QUESTION_WORDS = (
    "what",
    "how",
    "why",
    "when",
    "where",
    "who",
    "which",
)
_AUX_PREFIXES = (
    "is ",
    "are ",
    "do ",
    "does ",
    "can ",
    "will ",
)


def is_question(text: str) -> bool:
    """Return ``True`` when *text* is likely a question.

    The heuristic is intentionally simple and therefore robust.  It avoids
    pulling in sizeable NLP libraries that would increase cold-start time
    and memory footprint.  Three checks are performed (in order):

    1. The presence of a question-mark.
    2. The text starting with a standard WH-word (e.g. *what*, *how*…).
    3. An auxiliary verb commonly used to form questions (e.g. *is*, *do*).
    """

    text = text.strip().lower()
    return (
        "?" in text
        or text.startswith(_QUESTION_WORDS)
        or text.startswith(_AUX_PREFIXES)
    )



def contains_keywords(text: str, keywords: List[str]) -> bool:
    """Case-insensitive *any* keyword match."""

    tlower = text.lower()
    return any(kw.lower() in tlower for kw in keywords)



def truncate_history(
    history: List[Dict[str, Any]], max_items: int = 100
) -> List[Dict[str, Any]]:
    """Keep only the **last** *max_items* messages to bound context length."""

    return history[-max_items:]



def format_conversation_context(
    history: List[Dict[str, Any]], max_messages: int = 6
) -> str:
    """Serialize *history* into a compact plain-text context block."""

    if not history:
        return ""

    lines = (
        f"{item['role'].title()}: {item['content']}" for item in history[-max_messages:]
    )
    return "Previous conversation:\n" + "\n".join(lines)
