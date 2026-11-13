"""Load SQuAD 2.0 and SelfAware into a common Item, with deterministic balanced subsampling and splits."""
from __future__ import annotations

import hashlib
import math
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..config import resolve_path


@dataclass
class Item:
    id: str
    dataset: str
    question: str
    answerable: bool
    gold: list[str] = field(default_factory=list)      # acceptable answers (empty when unanswerable)
    context: str | None = None                          # SQuAD passage; None for SelfAware
    unanswerable_type: str | None = None                # "contextual" (SQuAD) | "inherent" (SelfAware) | None
    group: str = ""                                     # leakage group (SQuAD article title)
    source: str = ""
    split: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _h(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def _bucket(key: str, seed: int, fractions: dict[str, float]) -> str:
    u = (_h(seed, key) % 10_000) / 10_000
    acc = 0.0
    for name in ("train", "val", "test"):
        acc += float(fractions[name])
        if u < acc:
            return name
    return "test"


def _quotas(n: int, fractions: dict[str, float]) -> dict[str, int]:
    """Integer train/val/test counts that sum to n (largest-remainder, stable order)."""
    names = ("train", "val", "test")
    raw = {s: n * float(fractions[s]) for s in names}
    base = {s: int(math.floor(raw[s])) for s in names}
    leftover = n - sum(base.values())
    order = sorted(names, key=lambda s: (-(raw[s] - base[s]), names.index(s)))
    for i in range(leftover):
        base[order[i]] += 1
    return base


def assign_splits(items: list[Item], seed: int, fractions: dict[str, float]) -> None:
    """Deterministic split, stratified by answerability class.

    Items that share a group (a SQuAD passage) stay in the same split, so no passage is shared
    across splits. SelfAware groups are per-question, so class quotas are exact. Mixed-label
    groups (a passage with both answerable and unanswerable questions) are assigned as a unit;
    class balance is then as close as grouping allows.
    """
    names = ("train", "val", "test")
    groups: dict[str, list[Item]] = {}
    for it in items:
        groups.setdefault(it.group or it.id, []).append(it)

    def flags_of(members: list[Item]) -> set[bool]:
        return {bool(m.answerable) for m in members}

    n_class = {True: sum(bool(it.answerable) for it in items),
               False: sum(not it.answerable for it in items)}
    quota = {flag: _quotas(n, fractions) for flag, n in n_class.items()}
    filled = {flag: {s: 0 for s in names} for flag in (True, False)}
    assigned: dict[str, str] = {}

    def remaining(split: str, members: list[Item]) -> int:
        return sum(quota[bool(it.answerable)][split] - filled[bool(it.answerable)][split] for it in members)

    def place(gid: str) -> None:
        members = groups[gid]
        preferred = _bucket(gid, seed, fractions)
        # Prefer the split with the most remaining class quota; ties go to the hash bucket, then name order.
        split = max(names, key=lambda s: (remaining(s, members), 1 if s == preferred else 0, -names.index(s)))
        assigned[gid] = split
        for it in members:
            filled[bool(it.answerable)][split] += 1

    mixed, pure = [], {True: [], False: []}
    for gid, members in groups.items():
        fs = flags_of(members)
        if len(fs) == 1:
            pure[next(iter(fs))].append(gid)
        else:
            mixed.append(gid)
    for gid in sorted(mixed, key=lambda g: _h(seed, "split", g)):
        place(gid)
    for flag in (True, False):
        for gid in sorted(pure[flag], key=lambda g: _h(seed, "split", g)):
            place(gid)
    for it in items:
        it.split = assigned[it.group or it.id]


def balanced_subsample(items: list[Item], n_per_class: int, seed: int) -> list[Item]:
    out: list[Item] = []
    for flag in (True, False):
        pool = sorted((i for i in items if i.answerable is flag), key=lambda i: _h(seed, "sub", i.id))
        out.extend(pool[:n_per_class])
    return sorted(out, key=lambda i: i.id)


def load_squad_v2(path: str | Path, split: str = "dev") -> list[Item]:
    p = resolve_path(path)
    f = p / f"{split}-v2.0.json" if p.is_dir() else p
    data = json.loads(f.read_text())["data"]
    items: list[Item] = []
    for art in data:
        for pi, para in enumerate(art["paragraphs"]):
            for qa in para["qas"]:
                ans = not qa["is_impossible"]
                gold = sorted({a["text"] for a in qa.get("answers", [])}) if ans else []
                items.append(Item(id=f"squad_v2:{qa['id']}", dataset="squad_v2", question=qa["question"], answerable=ans,
                                  gold=gold, context=para["context"], unanswerable_type=None if ans else "contextual",
                                  group=f"{art['title']}#{pi}", source=f"squad_v2-{split}"))
    return items


def load_selfaware(path: str | Path) -> list[Item]:
    p = resolve_path(path)
    data = json.loads(p.read_text())["example"]
    items: list[Item] = []
    for ex in data:
        ans = bool(ex["answerable"])
        gold = [str(a) for a in (ex.get("answer") or [])] if ans else []
        items.append(Item(id=f"selfaware:{ex['question_id']}", dataset="selfaware", question=ex["question"], answerable=ans,
                          gold=gold, unanswerable_type=None if ans else "inherent", group=f"selfaware:{ex['question_id']}",
                          source=str(ex.get("source", ""))))
    return items


def load_dataset(name: str, dcfg: dict, seed: int, fractions: dict[str, float]) -> list[Item]:
    if name == "squad_v2":
        items = load_squad_v2(dcfg["path"], dcfg.get("split", "dev"))
    elif name == "selfaware":
        items = load_selfaware(dcfg["path"])
    else:
        raise ValueError(f"unknown dataset {name}")
    items = balanced_subsample(items, int(dcfg["n_per_class"]), seed)
    assign_splits(items, seed, fractions)
    return items


def load_all_items(cfg: dict) -> list[Item]:
    out: list[Item] = []
    for name, dcfg in cfg["datasets"].items():
        out.extend(load_dataset(name, dcfg, int(cfg["seed"]), cfg["splits"]))
    return out
