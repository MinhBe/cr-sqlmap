#!/usr/bin/env python3
"""
kaggle_runner.py - Path B automation for CR-sqlmap RL training on Kaggle.

RUN BY MINH, with his OWN Kaggle credentials. This script contains NO keys; it
only points KAGGLE_CONFIG_DIR at a per-account folder that holds that account's
kaggle.json, so it can rotate across the two accounts to spread the GPU budget
(~30h/session, ~60h/week combined).

One-time setup (Minh):
    pip install kaggle
    mkdir -p ~/.kaggle_accounts/withthecodebeone ~/.kaggle_accounts/minhart
    # put each account's kaggle.json in its folder, then:
    chmod 600 ~/.kaggle_accounts/*/kaggle.json
    # build the kernel dir (see RUN.md): ./kernel/ with kernel-metadata.json +
    # kaggle_waf_rl.py + circuit_breakers.py + fa2rnn.py + eval.py

Each invocation pushes+runs the kernel on the next account, polls to completion,
pulls outputs, and advances the rotation. Loop it (cron / `while true`) for an
unattended multi-day run.
"""
import json
import os
import subprocess
import sys
import time

ACCOUNTS = os.environ.get("KAGGLE_ACCOUNTS", "withthecodebeone,minhart").split(",")
ACCT_DIR = os.path.expanduser(os.environ.get("KAGGLE_ACCT_DIR", "~/.kaggle_accounts"))
KERNEL_DIR = os.environ.get("KERNEL_DIR", "./kernel")
KERNEL_SLUG = os.environ.get("KERNEL_SLUG", "cr-sqlmap-rl")
STATE = os.path.expanduser("~/.cr_sqlmap_runner_state.json")
POLL_S = int(os.environ.get("POLL_S", "300"))
MAX_POLLS = int(os.environ.get("MAX_POLLS", "360"))  # ~30h at 5-min polls


def _kaggle(acct, *args, timeout=900):
    env = dict(os.environ, KAGGLE_CONFIG_DIR=os.path.join(ACCT_DIR, acct))
    return subprocess.run(["kaggle", *args], env=env,
                          capture_output=True, text=True, timeout=timeout)


def _load_state():
    try:
        with open(STATE) as fh:
            return json.load(fh)
    except Exception:
        return {"idx": 0, "runs": []}


def _save_state(s):
    with open(STATE, "w") as fh:
        json.dump(s, fh, indent=2)


def main():
    s = _load_state()
    acct = ACCOUNTS[s["idx"] % len(ACCOUNTS)].strip()
    acct_cfg = os.path.join(ACCT_DIR, acct)
    if not os.path.exists(os.path.join(acct_cfg, "kaggle.json")):
        sys.exit(f"missing kaggle.json in {acct_cfg} (see RUN.md setup)")
    kernel = f"{acct}/{KERNEL_SLUG}"
    print(f"=== CR-sqlmap runner: account={acct} kernel={kernel} ===")

    push = _kaggle(acct, "kernels", "push", "-p", KERNEL_DIR)
    print("push:", push.stdout.strip() or push.stderr.strip())
    if push.returncode != 0:
        sys.exit("push failed (check kernel-metadata.json + kaggle.json)")

    for i in range(MAX_POLLS):
        st = _kaggle(acct, "kernels", "status", kernel).stdout.strip()
        print(time.strftime("%Y-%m-%d %H:%M"), st)
        low = st.lower()
        if any(k in low for k in ("complete", "error", "cancel")):
            break
        time.sleep(POLL_S)

    dest = os.path.join(os.environ.get("PULL_DIR", "./out"), acct)
    os.makedirs(dest, exist_ok=True)
    out = _kaggle(acct, "kernels", "output", kernel, "-p", dest)
    print("output ->", dest, out.stdout.strip() or out.stderr.strip())

    s["idx"] += 1
    s["runs"].append({"acct": acct, "kernel": kernel, "ts": time.time()})
    _save_state(s)
    nxt = ACCOUNTS[s["idx"] % len(ACCOUNTS)].strip()
    print(f"done; next invocation uses account: {nxt}")


if __name__ == "__main__":
    main()
