# CR-sqlmap — run it end to end

Authorized, isolated security-research lab. Two ways to run the training:
the **manual path** (simplest, recommended for your first hands-on check) and the
**automated Path B** (unattended, rotates both Kaggle accounts).

The pieces are already live on the Kali box and self-healing (cron keepalive +
@reboot): `relay.py` on `127.0.0.1:5000` behind an ngrok HTTPS tunnel. The
training runs on Kaggle and calls back to that relay.

---

## 0. Get the current relay URL + API key (on Kali)

```bash
ssh kali@192.168.244.129
cd ~/cr-sqlmap
./start_lab.sh                 # prints the current public ngrok URL
grep RELAY_API_KEY relay.env   # the X-API-Key value
```

The free ngrok URL changes if the tunnel restarts. If a run can't reach the
relay, re-read it here and update the Kaggle `RELAY_URL` secret. (A reserved
ngrok domain makes it permanent — optional.)

---

## Kaggle path — git clone (full circle)

1. New Notebook → **Add-ons → Secrets**: add `RELAY_URL` and `RELAY_API_KEY`
   (from step 0). For a **private** repo also add `GH_PAT` (a GitHub personal
   access token with `repo` scope).
2. **Settings**: Accelerator = GPU, Internet = ON.
3. First cell:
   ```python
   # public repo:
   !git clone https://github.com/MinhBe/cr-sqlmap.git
   # private repo instead:
   # from kaggle_secrets import UserSecretsClient
   # pat = UserSecretsClient().get_secret("GH_PAT")
   # !git clone https://{pat}@github.com/MinhBe/cr-sqlmap.git
   %cd cr-sqlmap
   !pip -q install -r requirements.txt
   !WAF=luawaf python kaggle_waf_rl.py   # sanity run; drop WAF=luawaf for the real one
   ```
4. Checkpoints + `eval_log.json` → `/kaggle/working/ckpt` (auto-resume next session).
   Evaluate: `!python eval.py run`.

Rotate accounts when you hit the ~30h cap: download `/kaggle/working/ckpt`,
re-upload to the other account's session, continue — or use `kaggle_runner.py`.

---

## Automated Path B (unattended, both accounts)

Run **on your machine with your own credentials** — `kaggle_runner.py` holds no
keys, it only switches which account folder it points at.

```bash
pip install kaggle
mkdir -p ~/.kaggle_accounts/withthecodebeone ~/.kaggle_accounts/minhart
# drop each account's kaggle.json into its folder:
chmod 600 ~/.kaggle_accounts/*/kaggle.json

# build the kernel dir
mkdir kernel && cp kaggle_waf_rl.py circuit_breakers.py fa2rnn.py eval.py kernel/
cp kernel-metadata.json kernel/
# edit kernel/kernel-metadata.json: set "id" to "<username>/cr-sqlmap-rl"
# add RELAY_URL + RELAY_API_KEY as Secrets on each account's kernel (Kaggle UI, once)

# one rotation (push → run → poll → pull → advance account):
python kaggle_runner.py
# unattended: loop it
while true; do python kaggle_runner.py; sleep 60; done
```

State is in `~/.cr_sqlmap_runner_state.json`; outputs land in `./out/<account>/`.

---

## Reward tunables (decided as defaults; retune after you see results)

- `R_RUL_SCALE=1000` — scales the APCT timing term. Raise/lower to change its
  weight vs. a success (R_SYS=10).
- `APCT_PROBES=3` — median probes per payload (timing-noise damping).
- γ breaker threshold `0.15` in `circuit_breakers.py` — see the code note on the
  one direction question to confirm against your doc.
- β schedule `0.15 → 0.90` in `kaggle_waf_rl.py`.

## R_REG (FA2RNN) — optional, lights up the regex reward term

Train once on Kaggle, then in the env set `env.fa2rnn = fa2rnn.FA2RNNReward(model)`.
Until then R_REG = 0 and R_SYS + R_RUL carry the reward (by design).

## Still open
- Eval **Groups B & D** are stubbed — their definitions weren't in any doc I
  could reach. Point me at the master doc and I'll wire them in.
