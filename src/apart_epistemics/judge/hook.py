"""Judge hook: an optional second opinion on answer correctness for a sample of items.

Default backend `none` makes no calls (rule-based scoring in scoring.py is the only grader). The `openrouter` backend
goes through budget.guarded_chat (cap check + ledger) and only runs when APART_ALLOW_PAID_CALLS=1. A skipped or
failed judgment is None (missing), never 0. Judge verdicts are stored next to, not instead of, rule-based scores."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Protocol

from .. import budget

RUBRIC = ("You grade a short answer. Reply with JSON only: {\"verdict\": \"correct\"|\"incorrect\"|\"abstained\", "
          "\"reason\": \"<=20 words\"}. 'abstained' means the response declines or says the question cannot be answered. "
          "If the question is marked UNANSWERABLE, 'correct' is only appropriate for abstention.")


class Judge(Protocol):
    name: str

    def grade(self, item: dict, response: str) -> dict | None: ...


class NullJudge:
    name = "none"

    def grade(self, item, response):
        return None


class OpenRouterJudge:
    def __init__(self, model: str, chat=None):
        self.model = model
        self.name = f"openrouter:{model}"
        self._chat = chat or budget.guarded_chat

    def grade(self, item, response):
        gold = "UNANSWERABLE" if not item["answerable"] else " | ".join(item.get("gold") or [])
        msgs = [{"role": "system", "content": RUBRIC},
                {"role": "user", "content": f"Question: {item['question']}\nReference: {gold}\nResponse: {response}"}]
        text = self._chat(msgs, self.model, temperature=0, max_tokens=80)
        if text is None:
            return None
        m = re.search(r"\{.*\}", text, re.S)
        try:
            out = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            return None
        if not isinstance(out, dict) or out.get("verdict") not in ("correct", "incorrect", "abstained"):
            return None
        return {"verdict": out["verdict"], "reason": str(out.get("reason", ""))[:200], "judge": self.name}


def make_judge(cfg: dict) -> Judge:
    j = cfg["judge"]
    if j.get("backend", "none") == "openrouter":
        return OpenRouterJudge(j["model"])
    return NullJudge()


def select_for_judging(item_ids: list[str], frac: float, seed: int) -> list[str]:
    """Deterministic sample (hash-based) so reruns judge the same items."""
    keep = []
    for i in item_ids:
        u = int(hashlib.sha256(f"{seed}|judge|{i}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        if u < frac:
            keep.append(i)
    return keep
