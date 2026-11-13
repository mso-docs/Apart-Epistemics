#!/usr/bin/env python3
"""Pre-release PII / private-link scan over text files (stdlib only).

    python3 scripts/pii_scan.py [PATH ...]        # default: the repo's tracked-style tree (src, configs, docs, scripts, tests, *.md)

Flags: absolute home paths, e-mail addresses, tailnet host names / 100.x addresses, private canvas/share links, key-like tokens,
and every line in .prepublish_identities (your real username, hostnames, ... one per line; git-ignored, never published).
Exit 1 when anything is found. Allowlisted: example.com addresses, 'user@host' placeholders.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "outputs", "outputs_synthetic", "data", "models", "logs", "hf_cache", "release", "node_modules"}
SKIP_SUFFIX = {".npz", ".npy", ".pt", ".bin", ".safetensors", ".png", ".jpg", ".pdf", ".sqlite3", ".pyc", ".egg-info"}
PATTERNS = {
    "home path": re.compile(r"(?:/Users|/home)/(?!user\b|you\b|USER\b|<)[A-Za-z0-9._-]+"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@(?!example\.(?:com|org)\b|host\b)[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "tailnet": re.compile(r"\b[a-z0-9-]+\.ts\.net\b|\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b"),
    "private link": re.compile(r"cursor\.com/canvas|docs\.google\.com/document/d/[A-Za-z0-9_-]{20,}/(?:edit|copy)\?|grok[- ]?bot", re.I),
    "token": re.compile(r"\b(?:hf_[A-Za-z0-9]{20,}|sk-or-(?:v1-)?[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})\b"),
}


# Apart's official public report template (linked from the sprint page); not private.
ALLOW = ("17b3JWki-ZkaqdmqGI5d0gof-HwlGuWU9Mc1_klNZXKs",)


def identities():
    f = ROOT / ".prepublish_identities"
    return [l.strip() for l in f.read_text().splitlines() if l.strip() and not l.startswith("#")] if f.exists() else []


def files(paths):
    for p in paths:
        p = Path(p)
        if p.is_file():
            yield p
            continue
        for q in sorted(p.rglob("*")):
            if q.is_file() and not (set(q.relative_to(p).parts[:-1]) & SKIP_DIRS) and q.suffix not in SKIP_SUFFIX and not q.name.endswith(".egg-info"):
                yield q


def main(argv):
    paths = argv or [ROOT / d for d in ("src", "configs", "docs", "scripts", "tests")] + list(ROOT.glob("*.md")) + list(ROOT.glob("*.toml")) + list(ROOT.glob("*.txt")) + [ROOT / ".env.example"]
    ids = identities()
    hits = []
    for f in files(paths):
        if f.resolve() == Path(__file__).resolve():
            continue
        try:
            text = f.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            for name, rx in PATTERNS.items():
                for m in rx.finditer(line):
                    if any(a in line for a in ALLOW):
                        continue
                    hits.append((f, n, name, m.group(0)))
            for ident in ids:
                if ident.lower() in line.lower():
                    hits.append((f, n, "identity", ident))
    for f, n, name, s in hits:
        print(f"{f.relative_to(ROOT) if f.is_relative_to(ROOT) else f}:{n}: {name}: {s}")
    print(f"{len(hits)} finding(s); {len(ids)} private identities checked")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
