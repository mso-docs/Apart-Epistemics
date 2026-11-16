#!/usr/bin/env python3
"""Download SQuAD 2.0 and SelfAware into data/raw/ (stdlib only; safe to run any time of day).

    python3 scripts/download_data.py            # download missing files, verify JSON parses
    python3 scripts/download_data.py --check    # only report what is present (no network)

Writes data/raw/SOURCES.json with URL, size, sha256 and license for each file.
Licenses: SQuAD 2.0 = CC BY-SA 4.0; SelfAware data = CC BY-SA 4.0 (repo code Apache 2.0). Attribute both in the report.
TriviaQA is optional and NOT downloaded here (license caveat is unresolved; see PROPOSAL.md).
"""
import argparse
import hashlib
import json
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
FILES = {
    "squad_v2/train-v2.0.json": ("https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json", "CC BY-SA 4.0", "SQuAD 2.0 (Rajpurkar et al., 2018)"),
    "squad_v2/dev-v2.0.json": ("https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json", "CC BY-SA 4.0", "SQuAD 2.0 (Rajpurkar et al., 2018)"),
    "selfaware/SelfAware.json": ("https://raw.githubusercontent.com/yinzhangyue/SelfAware/main/data/SelfAware.json", "CC BY-SA 4.0 (data); Apache 2.0 (code)", "SelfAware (Yin et al., Findings of ACL 2023)"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "apart-epistemics-download/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
    except (ssl.SSLError, urllib.error.URLError) as exc:
        # python.org builds on macOS ship without a CA bundle; curl uses the system keychain.
        print(f"  urllib failed ({type(exc).__name__}); falling back to curl", flush=True)
        subprocess.run(["curl", "-fsSL", "--retry", "3", "-o", str(tmp), url], check=True)
    json.loads(tmp.read_text())  # must be valid JSON before it replaces anything
    tmp.replace(dest)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    sources = {}
    ok = True
    for rel, (url, lic, name) in FILES.items():
        dest = RAW / rel
        if not args.check and (args.force or not dest.exists()):
            print(f"downloading {rel} ...", flush=True)
            fetch(url, dest)
        if dest.exists():
            size = dest.stat().st_size
            sources[rel] = {"name": name, "url": url, "license": lic, "bytes": size, "sha256": sha256(dest),
                            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(dest.stat().st_mtime))}
            print(f"ok  {rel}  {size / 1e6:.1f} MB")
        else:
            ok = False
            print(f"MISSING {rel}")
    if sources and not args.check:
        (RAW / "SOURCES.json").write_text(json.dumps(sources, indent=2) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
