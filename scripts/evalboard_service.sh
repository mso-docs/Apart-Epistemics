#!/bin/bash
# Install/refresh the Apart Evalboard as a persistent, LOOPBACK-ONLY launchd service on port 8767, with its own database
# (separate from the lexsi-unlearning project on 8765). Optionally expose it to the TAILNET ONLY via `tailscale serve`.
#   scripts/evalboard_service.sh install      # create db + layout, write launcher + plist, load it
#   scripts/evalboard_service.sh tailnet      # tailscale serve https :8444 -> 127.0.0.1:8767 (tailnet only; never funnel)
#   scripts/evalboard_service.sh status | stop | start
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUP="$HOME/Library/Application Support/evalboard"       # outside ~/Documents so launchd is not blocked by macOS TCC
DB="$SUP/data/projects/apart-epistemics/workspace.sqlite3"
PORT=8767; LABEL=local.apart-epistemics.evalboard; PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
PY=/Library/Frameworks/Python.framework/Versions/3.14/bin/python3
case "${1:-status}" in
  install)
    mkdir -p "$(dirname "$DB")"
    "$ROOT/.venv/bin/python" "$ROOT/scripts/setup_evalboard.py" --db "$DB"
    cat > "$SUP/run-apart.sh" <<RUN
#!/bin/bash
# Apart Evalboard launcher for launchd. Binds to 127.0.0.1 only.
set -euo pipefail
cd "$SUP"
export PYTHONUNBUFFERED=1
exec $PY -u -m evalboard.server --port $PORT --db "$DB"
RUN
    chmod +x "$SUP/run-apart.sh"
    cat > "$PLIST" <<P
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>$LABEL</string>
<key>WorkingDirectory</key><string>$SUP</string>
<key>ProgramArguments</key><array><string>$SUP/run-apart.sh</string></array>
<key>EnvironmentVariables</key><dict><key>PATH</key><string>/Library/Frameworks/Python.framework/Versions/3.14/bin:/usr/local/bin:/usr/bin:/bin</string><key>PYTHONUNBUFFERED</key><string>1</string></dict>
<key>KeepAlive</key><true/><key>RunAtLoad</key><true/>
<key>StandardOutPath</key><string>$HOME/Library/Logs/evalboard-$PORT.log</string>
<key>StandardErrorPath</key><string>$HOME/Library/Logs/evalboard-$PORT.log</string>
</dict></plist>
P
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$PLIST"
    sleep 2; curl -s -o /dev/null -w "evalboard (apart) http %{http_code} on 127.0.0.1:$PORT\n" "http://127.0.0.1:$PORT/api/workspace"
    ;;
  tailnet) tailscale serve --bg --https=8444 "http://127.0.0.1:$PORT"; tailscale serve status ;;
  stop) launchctl bootout "gui/$(id -u)/$LABEL" ;;
  start) launchctl bootstrap "gui/$(id -u)" "$PLIST" ;;
  status) launchctl list | grep "$LABEL" || echo "not loaded"; lsof -nP -iTCP:$PORT -sTCP:LISTEN || true; tailscale serve status ;;
esac
