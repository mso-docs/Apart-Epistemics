"""Prompt construction. The *plain* answer prompt has no abstention option: it measures committed behavior and is
the prompt whose last-token hidden state is probed. An optional abstain-option prompt measures what the model does
when refusal is explicitly allowed."""
from __future__ import annotations

ABSTAIN_TOKEN = "UNANSWERABLE"
SYSTEM = "You are a careful assistant. Answer concisely."


def answer_messages(item, abstain_option: bool = False) -> list[dict]:
    tail = f" If the question cannot be answered, reply exactly {ABSTAIN_TOKEN}." if abstain_option else ""
    if item.context is not None:
        user = (f"Answer the question using the passage. Reply with a short answer of a few words only.{tail}\n\n"
                f"Passage: {item.context}\n\nQuestion: {item.question}")
    else:
        user = f"Answer the question. Reply with a short answer of a few words only.{tail}\n\nQuestion: {item.question}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def confidence_messages(item, answer: str) -> list[dict]:
    base = answer_messages(item)
    return base + [
        {"role": "assistant", "content": answer},
        {"role": "user", "content": "How confident are you that your answer is correct? "
                                    "Reply with only a number from 0 (certain it is wrong) to 100 (certain it is right)."},
    ]
