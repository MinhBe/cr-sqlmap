#!/usr/bin/env bash
# start_lab.sh - idempotent keepalive for the CR-sqlmap relay + public tunnel.
# Tunnel backend selected by RELAY_TUNNEL (default: cloudflare). Prints the public URL.
# Single-host: relay + backend (modsec+DVWA on 127.0.0.1:8080) all on Kali; no sudo needed.
set -u
HOME_DIR="${HOME:-/home/kali}"
LAB="$HOME_DIR/cr-sqlmap"
NGROK="$HOME_DIR/bin/ngrok"
CFD="$HOME_DIR/bin/cloudflared"
cd "$LAB" || { echo "no $LAB"; exit 1; }
set -a; . ./relay.env 2>/dev/null; set +a
TUNNEL="${RELAY_TUNNEL:-cloudflare}"

# 1) relay on 127.0.0.1:5000
if ! pgrep -f "python3 relay.py" >/dev/null 2>&1; then
    nohup python3 relay.py > relay.log 2>&1 &
    sleep 2; echo "$(date -Is) started relay"
fi

# 2) public tunnel -> 5000
if [ "$TUNNEL" = "cloudflare" ]; then
    if ! pgrep -f "cloudflared tunnel" >/dev/null 2>&1; then
        if [ -n "${CF_TUNNEL_NAME:-}" ]; then
            # named tunnel = STABLE URL (needs prior `cloudflared tunnel login` + a domain/CNAME)
            nohup "$CFD" tunnel run "$CF_TUNNEL_NAME" > cloudflared.log 2>&1 &
        else
            # quick tunnel = random https://<x>.trycloudflare.com (no account needed)
            nohup "$CFD" tunnel --url http://localhost:5000 > cloudflared.log 2>&1 &
        fi
        sleep 8; echo "$(date -Is) started cloudflared"
    fi
    url="${CF_NAMED_URL:-}"
    [ -z "$url" ] && url=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" cloudflared.log 2>/dev/null | tail -1)
    echo "$(date -Is) relay=$(pgrep -f 'python3 relay.py'|tr '\n' ,) cloudflared=$(pgrep -f 'cloudflared tunnel'|tr '\n' ,) url=${url:-NONE}"
else
    if ! pgrep -f "ngrok http" >/dev/null 2>&1; then
        if [ -n "${NGROK_DOMAIN:-}" ]; then nohup "$NGROK" http --url="$NGROK_DOMAIN" 5000 > ngrok.log 2>&1 &
        else nohup "$NGROK" http 5000 > ngrok.log 2>&1 & fi
        sleep 6; echo "$(date -Is) started ngrok"
    fi
    url=$(curl -s http://127.0.0.1:4040/api/tunnels 2>/dev/null | python3 -c "import sys,json;d=json.load(sys.stdin);print((d.get('tunnels') or [{}])[0].get('public_url','NONE'))" 2>/dev/null)
    echo "$(date -Is) relay=$(pgrep -f 'python3 relay.py'|tr '\n' ,) ngrok=$(pgrep -f 'ngrok http'|tr '\n' ,) url=${url:-NONE}"
fi
