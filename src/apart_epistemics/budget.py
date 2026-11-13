"""Wiring to the standalone OpenRouter spend guard (openrouter_budget.py, project `apart`, $30 cap).

The guard module lives outside this repo; point APART_BUDGET_MODULE_DIR at its folder. Paid calls are OFF unless
APART_ALLOW_PAID_CALLS=1. Any judge call must go through `guarded_chat`, which checks the cap first and records spend.
A call skipped for budget reasons is MISSING data, never zero (callers get None).
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

DEFAULT_DIR = "~/Documents/code/tools/openrouter-budget"


class PaidCallsDisabled(RuntimeError):
    pass


def paid_calls_allowed() -> bool:
    return os.environ.get("APART_ALLOW_PAID_CALLS", "0") == "1"


def module_dir() -> Path:
    cands = [os.environ.get("APART_BUDGET_MODULE_DIR"), DEFAULT_DIR, "~/Documents/Code/tools/openrouter-budget"]
    for c in cands:
        if c and (Path(c).expanduser() / "openrouter_budget.py").is_file():
            return Path(c).expanduser()
    raise FileNotFoundError("openrouter_budget.py not found; set APART_BUDGET_MODULE_DIR")


def load_guard():
    d = str(module_dir())
    if d not in sys.path:
        sys.path.insert(0, d)
    return importlib.import_module("openrouter_budget")


def status() -> dict:
    """Spend so far vs cap, read from the ledger (no network)."""
    ob = load_guard()
    return {"cap_usd": ob.budget_usd(), "spent_usd": ob.spent_usd(), "remaining_usd": ob.remaining_usd(),
            "ledger": str(ob.ledger_path()), "module": str(module_dir()), "paid_calls_allowed": paid_calls_allowed()}


def guarded_chat(messages, model, **kw):
    """Budget-checked chat. Returns the reply text, or None when the cap is reached (treat as MISSING, not zero)."""
    if not paid_calls_allowed():
        raise PaidCallsDisabled("paid calls are disabled; set APART_ALLOW_PAID_CALLS=1 deliberately")
    ob = load_guard()
    try:
        body = ob.chat(messages, model=model, **kw)
    except ob.BudgetExceeded:
        return None
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
