#!/usr/bin/env bash
# start_lab.sh - idempotent keepalive for the CR-sqlmap relay + ngrok tunnel.
# Safe to run repeatedly (cron / @reboot / by hand): starts whatever is down,
# leaves what's up, and prints the current public URL. No sudo required.
set -u
HOME_DIR="${HOME:-/home/kali}"
LAB="$HOME_DIR/cr-sqlmap"
NGROK="$HOME_DIR/bin/ngrok"
cd "$LAB" || { echo "no $LAB"; exit 1; }

# load config (RELAY_* + optional NGROK_DOMAIN for a stable URL)
set -a; . ./relay.env 2>/dev/null; set +a

# 1) relay on 127.0.0.1:5000
if ! pgrep -f "python3 relay.py" >/dev/null 2>&1; then
    nohup python3 relay.py > relay.log 2>&1 &
    sleep 2
    echo "$(date -Is) started relay"
fi

# 2) ngrok tunnel -> 5000. With a reserved domain (NGROK_DOMAIN in relay.env) the
# public URL is STABLE across restarts; otherwise the free tier rotates it.
if ! pgrep -f "ngrok http" >/dev/null 2>&1; then
    if [ -n "${NGROK_DOMAIN:-}" ]; then
        nohup "$NGROK" http --url="$NGROK_DOMAIN" 5000 > ngrok.log 2>&1 &
    else
        nohup "$NGROK" http 5000 > ngrok.log 2>&1 &
    fi
    sleep 6
    echo "$(date -Is) started ngrok"
fi

# 3) report current public URL (ngrok local API)
url=$(curl -s http://127.0.0.1:4040/api/tunnels 2>/dev/null | \
      python3 -c "import sys,json;d=json.load(sys.stdin);print((d.get('tunnels') or [{}])[0].get('public_url','NONE'))" 2>/dev/null)
echo "$(date -Is) relay_pid=$(pgrep -f 'python3 relay.py' | tr '\n' ',') ngrok_pid=$(pgrep -f 'ngrok http' | tr '\n' ',') url=${url:-NONE}"
