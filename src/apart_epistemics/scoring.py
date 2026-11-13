"""Answer scoring: SQuAD-style EM/F1, SelfAware answer-string match, abstention detection, confidence parsing."""
from __future__ import annotations

import re
import string
from collections import Counter

from .prompts import ABSTAIN_TOKEN

_ARTICLES = re.compile(r"\b(a|an|the)\b")
ABSTAIN_PATTERNS = [
    r"\bunanswerable\b", r"\bcannot be answered\b", r"\bcan(?:'|’)?t be answered\b", r"\bno answer\b",
    r"\bnot (?:enough|sufficient) information\b", r"\bnot (?:mentioned|stated|provided|specified)\b",
    r"\bi (?:do not|don(?:'|’)?t) know\b", r"\bi(?:'|’)?m not sure\b", r"\bi am not sure\b", r"\bunknown\b",
    r"\bit is not possible to (?:answer|determine|know)\b", r"\bthere is no (?:definitive|single|way)\b",
    r"\bno one knows\b", r"\bimpossible to (?:answer|say|know|determine)\b",
]
_ABSTAIN_RE = re.compile("|".join(ABSTAIN_PATTERNS), re.I)


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds: list[str]) -> bool:
    p = normalize(pred)
    return any(p == normalize(g) for g in golds)


def f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


def best_f1(pred: str, golds: list[str]) -> float:
    return max((f1(pred, g) for g in golds), default=0.0)


def string_match(pred: str, golds: list[str]) -> bool:
    """SelfAware answers are short free text: accept if a gold answer is contained in the prediction (or vice versa)."""
    p = normalize(pred)
    if not p:
        return False
    for g in golds:
        gn = normalize(g)
        if gn and (gn == p or (len(gn) >= 3 and gn in p) or (len(p) >= 3 and p in gn)):
            return True
    return False


def is_abstention(text: str) -> bool:
    t = (text or "").strip()
    return bool(t) and (ABSTAIN_TOKEN.lower() in t.lower() or _ABSTAIN_RE.search(t) is not None)


def score_answer(item, response: str, f1_threshold: float = 0.5) -> dict:
    """Return {abstained, correct, f1}. `correct` is None when abstained on an answerable item (neither right nor wrong
    answer text), False when an unanswerable item is answered, True when an unanswerable item is abstained on."""
    abst = is_abstention(response)
    if not item.answerable:
        return {"abstained": abst, "correct": bool(abst), "f1": None}
    if abst:
        return {"abstained": True, "correct": False, "f1": 0.0}
    if item.dataset == "squad_v2":
        sc = best_f1(response, item.gold)
        return {"abstained": False, "correct": bool(exact_match(response, item.gold) or sc >= f1_threshold), "f1": sc}
    ok = string_match(response, item.gold)
    return {"abstained": False, "correct": bool(ok), "f1": None}


_NUM = re.compile(r"(?<![\d.])(\d{1,4}(?:\.\d+)?)(?![\d])\s*%?")


def parse_confidence(text: str) -> float | None:
    """Parse a 0-100 verbalized confidence to [0, 1]; None when absent or out of range (missing, not zero)."""
    m = _NUM.search(text or "")
    if not m:
        return None
    v = float(m.group(1))
    if v > 100:
        return None
    return v / 100.0
