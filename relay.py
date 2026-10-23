#!/usr/bin/env python3
"""
relay.py - Flask relay for the CR-sqlmap RL lab (isolated lab use only).

Exposes executor.run_episode() over HTTP so the Kaggle training notebook can
POST an action and get back the sqlmap telemetry as JSON. Meant to sit behind an
ngrok HTTPS tunnel (ngrok -> 127.0.0.1:RELAY_PORT).

Safety / abuse controls (the ngrok URL is public):
  * Targets are an ALLOWLIST of the two local lab WAF endpoints only. The caller
    picks `waf` ("modsec" | "luawaf"); it can never supply an arbitrary URL, so
    the endpoint cannot be used to attack anything but the lab.
  * Every /run needs the shared secret in the `X-API-Key` header.
  * Episodes are serialized (one sqlmap run at a time) to avoid overload.

Env:
  RELAY_API_KEY  shared secret (required to protect /run)
  RELAY_PORT     listen port on 127.0.0.1 (default 5000)
"""

import os
import threading

from flask import Flask, jsonify, request

import executor
import libinj_oracle

API_KEY = os.environ.get("RELAY_API_KEY", "")

# Single-host on Kali: backend (modsec+DVWA) runs locally on 127.0.0.1:8080.
# (Was 192.168.244.128 = Ubuntu over LAN; Ubuntu dropped in the single-host move.)
TARGET = os.environ.get("TARGET_HOST", "127.0.0.1")
TARGETS = {
    "modsec": {  # ModSecurity + OWASP-CRS v3.2.0 (REQUEST-942 only)
        "base": f"http://{TARGET}:8080",
        "url":  f"http://{TARGET}:8080/vulnerabilities/sqli/?id=1&Submit=Submit",
    },
    "luawaf": {  # ngx_lua_waf
        "base": f"http://{TARGET}:8081",
        "url":  f"http://{TARGET}:8081/vulnerabilities/sqli/?id=1&Submit=Submit",
    },
}

app = Flask(__name__)
# Bounded concurrency: allow up to RELAY_MAX_CONCURRENT episodes at once (default 1
# = serial, current behaviour). Raise it (e.g. 4) for parallel-env training once the
# Kali VM has RAM/CPU headroom. The APCT timing probe is serialized inside executor
# so the timing side-channel stays clean even under concurrent scans.
_MAX_CONCURRENT = int(os.environ.get("RELAY_MAX_CONCURRENT", "1"))
_sem = threading.BoundedSemaphore(_MAX_CONCURRENT)


@app.get("/health")
def health():
    return jsonify(status="ok", targets=list(TARGETS), auth_required=bool(API_KEY),
                   libinj=True)


@app.post("/libinj")
def libinj():
    """Offline libinjection oracle (no WAF round-trip). Body: {"payload": "..."} or
    {"payloads": ["...", ...]}. Returns per-payload {sqli, fingerprint}. This is the
    dense local signal (R_LIB / GA fitness): sqli=False => fingerprint left libinjection's
    bad-set => candidate to slip CRS rule 942100."""
    if API_KEY and request.headers.get("X-API-Key") != API_KEY:
        return jsonify(error="unauthorized"), 401
    body = request.get_json(force=True, silent=True) or {}
    payloads = body.get("payloads")
    if payloads is None:
        one = body.get("payload")
        if one is None:
            return jsonify(error="provide 'payload' or 'payloads'"), 400
        payloads = [one]
    results = [dict(payload=p, **libinj_oracle.verdict(p)) for p in payloads]
    return jsonify(results=results)


@app.post("/run")
def run():
    if API_KEY and request.headers.get("X-API-Key") != API_KEY:
        return jsonify(error="unauthorized"), 401

    body = request.get_json(force=True, silent=True) or {}
    waf = body.get("waf", "modsec")
    if waf not in TARGETS:
        return jsonify(error=f"unknown waf '{waf}'", choices=list(TARGETS)), 400
    tgt = TARGETS[waf]

    tamper = body.get("tamper", [])
    if isinstance(tamper, str):
        tamper = [t for t in tamper.split(",") if t]
    extra = body.get("extra", [])
    if isinstance(extra, str):
        extra = extra.split()

    action = dict(
        target=tgt["url"],
        login_base=tgt["base"],
        param=body.get("param", "id"),
        tamper=tamper,
        technique=body.get("technique"),
        level=int(body.get("level", 1)),
        risk=int(body.get("risk", 1)),
        random_agent=bool(body.get("random_agent", False)),
        extra=extra,
        overall_timeout=int(body.get("timeout", 180)),
        security=body.get("security", "low"),
    )

    if not _sem.acquire(blocking=False):
        return jsonify(error=f"busy: {_MAX_CONCURRENT} concurrent episode(s) in flight"), 429
    try:
        result = executor.run_episode(**action)
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        return jsonify(error=f"executor failed: {exc}", waf=waf), 500
    finally:
        _sem.release()

    result["waf"] = waf
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("RELAY_PORT", "5000"))
    # 127.0.0.1 only: the public ingress is ngrok, never the LAN directly.
    app.run(host="127.0.0.1", port=port, threaded=True)
