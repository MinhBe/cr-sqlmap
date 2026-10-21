# CR-sqlmap

RL agent that drives real `sqlmap` to detect SQL injection and learn to evade
WAFs, in an **isolated, authorized security-research lab**. Fusion of Checklist-RL
(Paper 1) and Ouroboros (Paper 2) with an annealed-hybrid reward, a symbolic
FA2RNN reward term, and reward-hacking / compute circuit breakers.

> ⚠️ Authorized lab use only. The target VM stays LAN-only; nothing here is for
> use against systems you don't own or have permission to test.

## The full circle

```
Kaggle (RL training, GPU)
   │  HTTPS (ngrok)            POST /run {action}  ->  JSON telemetry
   ▼
Kali VM: relay.py (Flask, :5000)  ──►  executor.py  ──►  sqlmap
   │                                                       │ LAN
   ▼                                                       ▼
Ubuntu VM: nginx+ModSecurity/CRS-942 (:8080)  &  ngx_lua_waf (:8081)  ──►  DVWA
```

The agent (Kaggle) picks an action (tamper scripts, technique, level, risk,
random-agent), the relay runs one sqlmap episode on Kali against the WAF-fronted
DVWA, and returns telemetry; the reward blends true evasion success (`R_SYS`), a
timing side-channel (`R_RUL`), and the FA2RNN regex term (`R_REG`).

## Files

| file | role |
|---|---|
| `executor.py` | sqlmap wrapper: action → sqlmap → JSON telemetry (+ χ_ORIG/χ_MUT APCT probe) |
| `relay.py` | Flask relay (Kali), API-key auth, lab-only target allowlist |
| `kaggle_waf_rl.py` | Gym env + DQN (stable-baselines3) + annealed reward + gold-eval + breakers |
| `circuit_breakers.py` | anti-reward-hacking (γ) + anti-compute-runaway |
| `fa2rnn.py` | FA-RNN for `R_REG` (regex → weighted-FA → linear-RNN + MLP → ψ) |
| `eval.py` | metric Groups A–D (TPR/FPR/F1, transfer matrix, payload diversity, γ/ρₑ) |
| `kaggle_runner.py` | optional two-account Kaggle rotation (run locally with your creds) |
| `start_lab.sh` | Kali keepalive: relay + ngrok (idempotent) |
| `kernel-metadata.json` | Kaggle kernel metadata (CLI path) |

## 1. Bring up the Kali lab

```bash
# on Kali (sqlmap, python3, flask already present)
cd ~/cr-sqlmap
./start_lab.sh                 # starts relay + ngrok, prints the public URL
grep RELAY_API_KEY relay.env   # your relay API key (kept out of git)
```

`start_lab.sh` is wired to cron (`@reboot` + every 5 min) so relay + ngrok
self-heal. The ngrok free-tier URL changes on restart — re-read it with
`./start_lab.sh` and update the Kaggle `RELAY_URL` secret if it moves.

Secrets (`relay.env`, ngrok authtoken, SSH keys) live only on the box and are
`.gitignore`d — never in this repo.

## 2. Run the training on Kaggle (git-clone, full circle)

New Notebook → **Add-ons → Secrets**: add `RELAY_URL` (the ngrok URL) and
`RELAY_API_KEY` (from step 1). **Settings:** Accelerator = GPU, Internet = ON.

First cell:

```python
# public repo:
!git clone https://github.com/MinhBe/cr-sqlmap.git
# private repo: store a GitHub PAT as a Kaggle secret GH_PAT, then:
# from kaggle_secrets import UserSecretsClient
# pat = UserSecretsClient().get_secret("GH_PAT")
# !git clone https://{pat}@github.com/MinhBe/cr-sqlmap.git

%cd cr-sqlmap
!pip -q install -r requirements.txt
# first sanity run against the weaker WAF (fast successes):
!WAF=luawaf python kaggle_waf_rl.py
# the real experiment (hard target):
# !python kaggle_waf_rl.py
```

Checkpoints + `eval_log.json` land in `/kaggle/working/ckpt` (auto-resume next
session). Evaluate: `!python eval.py run`.

Unattended two-account rotation: `kaggle_runner.py` (see its header), run on your
own machine with your own `kaggle.json` per account.

## Reward tunables (defaults; retune after results)

- β surrogate weight anneals **0.90 → 0.15** (reduce surrogate dependence over
  training = anti-reward-hacking, master doc A.6).
- `R_RUL_SCALE=1000`, `APCT_PROBES=3`, γ threshold `0.15`.
- `GOLD_EVAL_EVERY=2000`, `GOLD_EVAL_K=10`.

## Secrets policy

No secrets in git, ever. `RELAY_URL` / `RELAY_API_KEY` → Kaggle Secrets.
GitHub PAT (if private) → Kaggle Secret `GH_PAT`. See `.gitignore`.
