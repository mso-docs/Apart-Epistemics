#!/usr/bin/env python3
"""Create/refresh the Apart Evalboard project layout (no data is loaded).

    python3 scripts/setup_evalboard.py --db "<.../projects/apart-epistemics/workspace.sqlite3>"   # initialise the DB + layout
    python3 scripts/setup_evalboard.py --url http://127.0.0.1:8767 [--force]                       # apply to a running server

An existing custom layout (explicit pages) is preserved unless --force. The loopback-only server for this project is
started by scripts/evalboard_service.sh (launchd, port 8767, tailnet via `tailscale serve`).
"""
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db")
    ap.add_argument("--url")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--layout-file", default=str(ROOT / "configs/evalboard_layout.json"))
    a = ap.parse_args(argv)
    layout = {k: v for k, v in json.loads(Path(a.layout_file).read_text()).items() if not k.startswith("_")}
    if a.db:
        for c in (os.environ.get("EVALBOARD_ROOT"), Path.home() / "Documents/Code/modular_dashboard"):
            if c and (Path(c) / "evalboard/core.py").is_file():
                sys.path.insert(0, str(c)); break
        from evalboard.core import Store
        store = Store(Path(a.db))
        cur = store.snapshot().get("layout") or {}
        if cur.get("pages") and not a.force:
            print("layout already has custom pages; kept (use --force to replace)"); return 0
        saved = store.save_layout({**layout})
        print("layout applied; primary metric:", [m["key"] for m in saved.get("metrics", []) if m.get("primary")])
        return 0
    if not a.url:
        ap.error("--db or --url required")
    if urllib.parse.urlsplit(a.url).hostname not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit("loopback URLs only")
    cur = json.load(urllib.request.urlopen(a.url + "/api/workspace", timeout=30)).get("layout") or {}
    if cur.get("pages") and not a.force:
        print("layout already has custom pages; kept"); return 0
    req = urllib.request.Request(a.url + "/api/layout", json.dumps(layout).encode(), {"Content-Type": "application/json"})
    saved = json.load(urllib.request.urlopen(req, timeout=30))
    print("layout applied; primary metric:", [m["key"] for m in saved.get("metrics", []) if m.get("primary")])
    return 0


if __name__ == "__main__":
    sys.exit(main())
