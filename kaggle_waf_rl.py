#!/usr/bin/env python3
# %% [markdown]
# CR-sqlmap — Phase 5: RL training notebook (Kaggle)
# -------------------------------------------------
# DQN agent that learns to evade WAFs by driving the Kali `executor.py` through
# the `relay.py` HTTPS endpoint (ngrok). Isolated authorized lab only.
#
# Reward (Ouroboros base + Minh's annealed hybrid):
#   R_SYS  = 10 if evasion succeeds else 0                         (live)
#   R_RUL  = APCT_mut - APCT_orig  (timing side-channel)           (stub -> 0)
#   R_REG  = F_ARNN(orig) - F_ARNN(mut)  (FA2RNN, Tier-4)          (stub -> 0)
#   R_SYN  = clip(R_SYS + R_RUL + R_REG, 0, 10)       (unweighted sum)
#   R_tot  = (1-beta)*R_outcome + beta*R_SYN,  beta annealed 0.15 -> 0.90
#
# The relay client + action decode + reward + SoftChecklist are plain-Python so
# they can be unit/smoke-tested without torch/gym (see `smoke` at the bottom).
# Heavy deps (gym, stable-baselines3, torch, transformers) are imported lazily
# inside the training section only.

# %%
import json
import math
import os
import time
import urllib.error
import urllib.request

# ----------------------------------------------------------------------------
# CONFIG  (on Kaggle set these via Secrets / notebook params)
# ----------------------------------------------------------------------------
def _secret(name, default=""):
    """Resolve config from env first, then Kaggle Secrets (UserSecretsClient)."""
    v = os.environ.get(name)
    if v:
        return v
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return default


RELAY_URL = _secret("RELAY_URL", "https://4f6c-1-55-175-177.ngrok-free.app")
# NOTE: free ngrok URL is ephemeral; update the RELAY_URL secret each session or
# use a reserved domain. Current URL: ssh kali 'curl -s http://127.0.0.1:4040/api/tunnels'
RELAY_API_KEY = _secret("RELAY_API_KEY", "")  # Kaggle Secret or env var
EPISODE_TIMEOUT = int(os.environ.get("EPISODE_TIMEOUT", "180"))
WAF = os.environ.get("WAF", "modsec")  # which target this run trains against

ALGO = os.environ.get("ALGO", "DQN")   # DQN | PPO  (one-line swap)
GAMMA = float(os.environ.get("GAMMA", "0.8"))
TOTAL_TIMESTEPS = int(os.environ.get("TOTAL_TIMESTEPS", "20000"))
EP_LEN = int(os.environ.get("EP_LEN", "32"))         # steps per episode
# beta = weight on the dense surrogate R_SYN. Per master doc (A.6, §reward): anneal
# DOWN over training so surrogate dependence falls and the true outcome dominates
# late = anti-reward-hacking. Early: surrogate-heavy bootstrap; late: outcome-heavy.
BETA_START, BETA_END = 0.90, 0.15
CHECKLIST_ALPHA = 0.30
USE_BERT = os.environ.get("USE_BERT", "1") == "1"
CKPT_DIR = os.environ.get("CKPT_DIR", "/kaggle/working/ckpt")
# C.6 gold-eval loop: freeze policy every GOLD_EVAL_EVERY steps, take GOLD_EVAL_K
# deterministic (eps=0) attempts, record true TPR. Gated small so the long run's
# compute budget (and the anti-runaway breaker) are unaffected.
GOLD_EVAL_EVERY = int(os.environ.get("GOLD_EVAL_EVERY", "2000"))
GOLD_EVAL_K = int(os.environ.get("GOLD_EVAL_K", "10"))
LEARNING_STARTS = int(os.environ.get("LEARNING_STARTS", "200"))  # lower for quick sanity
DEBUG_EPISODES = os.environ.get("DEBUG_EPISODES", "1") == "1"    # live per-episode logging

# ----------------------------------------------------------------------------
# ACTION SPACE — 9 x 3 x 2 x 7 x 2 = 756 discrete actions
# (filling in the technique/operator presets the master doc left open — adjust freely)
# ----------------------------------------------------------------------------
TECHNIQUES = ["B", "E", "U", "S", "T", "Q", "BE", "BEU", "BEUSTQ"]        # 9
LEVELS = [1, 2, 3]                                                         # 3
RISKS = [1, 2]                                                             # 2
OPERATORS = [                                                              # 7 tamper presets
    [],                                              # 0: no tamper
    ["space2comment"],                               # 1
    ["between"],                                     # 2
    ["randomcase"],                                  # 3
    ["charencode"],                                  # 4
    ["space2comment", "between", "randomcase"],      # 5
    ["apostrophemask", "equaltolike", "between"],    # 6
]
RANDOM_AGENT = [False, True]                                               # 2
N_ACTIONS = len(TECHNIQUES) * len(LEVELS) * len(RISKS) * len(OPERATORS) * len(RANDOM_AGENT)
assert N_ACTIONS == 756, N_ACTIONS


def decode_action(idx):
    """Map a discrete action index in [0,756) to concrete sqlmap params."""
    i = int(idx)
    i, ra = divmod(i, len(RANDOM_AGENT))
    i, op = divmod(i, len(OPERATORS))
    i, rk = divmod(i, len(RISKS))
    i, lv = divmod(i, len(LEVELS))
    i, te = divmod(i, len(TECHNIQUES))
    return {
        "technique": TECHNIQUES[te],
        "level": LEVELS[lv],
        "risk": RISKS[rk],
        "tamper": OPERATORS[op],
        "random_agent": RANDOM_AGENT[ra],
        "_idx": {"technique": te, "level": lv, "risk": rk, "operator": op, "random_agent": ra},
    }

# ----------------------------------------------------------------------------
# RELAY CLIENT
# ----------------------------------------------------------------------------
def call_kali(action, waf=WAF, timeout=EPISODE_TIMEOUT, retries=2):
    """POST one action to the relay and return the executor telemetry dict."""
    body = {
        "waf": waf,
        "tamper": action["tamper"],
        "technique": action["technique"],
        "level": action["level"],
        "risk": action["risk"],
        "random_agent": action["random_agent"],
        "timeout": timeout,
    }
    data = json.dumps(body).encode()
    headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true",
    }
    if RELAY_API_KEY:
        headers["X-API-Key"] = RELAY_API_KEY
    last = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(RELAY_URL + "/run", data=data,
                                     headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout + 15) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001 - any failure -> visible, then retry
            last = e
            print(f"[relay] call failed (attempt {attempt+1}/{retries+1}): {e}", flush=True)
            time.sleep(2 * (attempt + 1))
    print(f"[relay] UNREACHABLE after {retries+1} tries -> {RELAY_URL} ({last}); "
          f"returning empty episode (found=False)", flush=True)
    return {"found": False, "requests": 0, "blocked_responses": 0,
            "block_ratio": 0.0, "duration_s": 0.0, "timed_out": True,
            "error": f"relay unreachable: {last}", "techniques": [],
            "dbms": None, "payloads": [], "waf_warned": False,
            "not_injectable": True}


def preflight():
    """Fail fast with a clear message if the relay/tunnel is unreachable, instead
    of silently crawling through per-step timeouts."""
    url = RELAY_URL.rstrip("/") + "/health"
    try:
        req = urllib.request.Request(url, headers={"ngrok-skip-browser-warning": "true"})
        with urllib.request.urlopen(req, timeout=12) as r:
            print(f"[preflight] relay OK at {RELAY_URL}: {r.read().decode()}", flush=True)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"[preflight] CANNOT reach relay at {RELAY_URL!r} ({e}). On Kali run "
            f"./start_lab.sh to get the CURRENT ngrok URL, set the Kaggle Secret "
            f"RELAY_URL to exactly that, confirm Internet=ON, then rerun.") from e

# ----------------------------------------------------------------------------
# SOFT CHECKLIST — EMA over 9 "blocked keyword" buckets (Paper-1 soft version)
# NOTE: executor currently reports aggregate block_ratio, not per-keyword blocks.
# Until the executor emits blocked-payload keywords (a Phase 6 enhancement), the
# buckets are seeded from block_ratio as a placeholder; the EMA machinery and the
# state slot are in place so the richer signal drops in without a shape change.
# ----------------------------------------------------------------------------
CHECKLIST_KEYS = ["union", "select", "sleep", "or", "and",
                  "concat", "information_schema", "benchmark", "comment"]

class SoftChecklist:
    def __init__(self, alpha=CHECKLIST_ALPHA):
        self.alpha = alpha
        self.ema = [0.0] * len(CHECKLIST_KEYS)

    def update(self, telemetry):
        br = float(telemetry.get("block_ratio", 0.0))
        payloads = " ".join(telemetry.get("payloads", [])).lower()
        for i, k in enumerate(CHECKLIST_KEYS):
            # signal: keyword present in a surviving payload lowers its "blocked" EMA
            present_and_passed = 1.0 if (k in payloads and telemetry.get("found")) else br
            self.ema[i] = (1 - self.alpha) * self.ema[i] + self.alpha * present_and_passed
        return list(self.ema)

# ----------------------------------------------------------------------------
# REWARD COMBINER
# ----------------------------------------------------------------------------
class RewardCombiner:
    """Ouroboros R_SYN + Minh's annealed hybrid. beta (surrogate weight) anneals
    DOWN (0.90->0.15): early training leans on the dense surrogate R_SYN to
    bootstrap, then shifts onto the true outcome R_SYS as beta falls - reducing
    surrogate dependence over training to resist reward-hacking (master doc A.6)."""

    def __init__(self, total_timesteps=TOTAL_TIMESTEPS,
                 beta_start=BETA_START, beta_end=BETA_END):
        self.total = max(1, total_timesteps)
        self.b0, self.b1 = beta_start, beta_end
        self._t = 0

    def beta(self):
        frac = min(1.0, self._t / self.total)
        return self.b0 + (self.b1 - self.b0) * frac

    @staticmethod
    def r_sys(tele):
        return 10.0 if (tele.get("found") and not tele.get("timed_out")) else 0.0

    # R_RUL_SCALE: APCT is seconds/char so raw deltas are ~1e-4. Decision (Minh
    # delegated): default 1000.0 so a typical APCT delta lands ~0.1-1.0, a few % of
    # a success (R_SYS=10) without dominating it. Minh to hand-tune after results.
    R_RUL_SCALE = float(os.environ.get("R_RUL_SCALE", "1000.0"))

    @classmethod
    def r_rul(cls, tele):
        # Eq6: APCT_mut - APCT_orig, from the executor's chi_ORIG/chi_MUT probe.
        pr = tele.get("probe") or {}
        if "apct_mut" in pr and "apct_orig" in pr:
            return cls.R_RUL_SCALE * float(pr["apct_mut"] - pr["apct_orig"])
        return 0.0

    @staticmethod
    def r_reg(tele, fa2rnn=None):
        # Eq7: psi(chi_ORIG) - psi(chi_MUT) from the trained FA2RNN. 0 until a model
        # is supplied; the probe provides the orig/mut strings.
        pr = tele.get("probe") or {}
        if fa2rnn is None or "orig" not in pr or "mut" not in pr:
            return 0.0
        return float(fa2rnn.delta(pr["orig"], pr["mut"]))

    def compute(self, tele, fa2rnn=None):
        r_sys = self.r_sys(tele)
        r_rul = self.r_rul(tele)
        r_reg = self.r_reg(tele, fa2rnn)
        r_syn = max(0.0, min(10.0, r_sys + r_rul + r_reg))
        r_outcome = r_sys  # pure sparse outcome
        beta = self.beta()
        r_total = (1 - beta) * r_outcome + beta * r_syn
        self._t += 1
        return r_total, {"R_SYS": r_sys, "R_RUL": r_rul, "R_REG": r_reg,
                         "R_SYN": r_syn, "R_outcome": r_outcome, "beta": beta}

# ----------------------------------------------------------------------------
# ENGINEERED STATE — 15 features (+ 9 SoftChecklist EMA = 24), then BERT(768)
# ----------------------------------------------------------------------------
MAX_REQ = 4000.0

def engineered_state(tele, action, waf, checklist_ema):
    idx = action["_idx"]
    feats = [
        1.0 if tele.get("found") else 0.0,                       # 1
        float(tele.get("block_ratio", 0.0)),                     # 2
        min(1.0, tele.get("requests", 0) / MAX_REQ),             # 3
        min(1.0, tele.get("duration_s", 0.0) / EPISODE_TIMEOUT), # 4
        1.0 if tele.get("waf_warned") else 0.0,                  # 5
        1.0 if tele.get("not_injectable") else 0.0,              # 6
        1.0 if tele.get("dbms") else 0.0,                        # 7
        len(tele.get("techniques", [])) / 6.0,                   # 8
        1.0 if tele.get("timed_out") else 0.0,                   # 9
        idx["technique"] / (len(TECHNIQUES) - 1),                # 10
        idx["level"] / (len(LEVELS) - 1),                        # 11
        idx["risk"] / (len(RISKS) - 1),                          # 12
        idx["operator"] / (len(OPERATORS) - 1),                  # 13
        float(idx["random_agent"]),                              # 14
        0.0 if waf == "modsec" else 1.0,                         # 15
    ]
    return feats + list(checklist_ema)  # 15 + 9 = 24


class BertEncoder:
    """Lazy BERT CLS encoder (768-d) of a text summary of the last episode."""
    def __init__(self, enabled=USE_BERT):
        self.enabled = enabled
        self._tok = self._model = None

    def _lazy(self):
        if self._model is None:
            import torch  # noqa
            from transformers import AutoModel, AutoTokenizer
            self._torch = torch
            self._dev = "cuda" if torch.cuda.is_available() else "cpu"
            self._tok = AutoTokenizer.from_pretrained("bert-base-uncased")
            self._model = AutoModel.from_pretrained("bert-base-uncased").eval().to(self._dev)

    def encode(self, tele, action, waf):
        if not self.enabled:
            return [0.0] * 768
        self._lazy()
        text = (f"waf={waf} tech={action['technique']} level={action['level']} "
                f"risk={action['risk']} tamper={','.join(action['tamper']) or 'none'} "
                f"ra={action['random_agent']} found={tele.get('found')} "
                f"dbms={tele.get('dbms')} block_ratio={tele.get('block_ratio')} "
                f"reqs={tele.get('requests')} techniques={tele.get('techniques')}")
        t = self._tok(text, return_tensors="pt", truncation=True, max_length=128).to(self._dev)
        with self._torch.no_grad():
            out = self._model(**t)
        return out.last_hidden_state[:, 0, :].squeeze(0).cpu().tolist()  # CLS, 768 (GPU if avail)

# ----------------------------------------------------------------------------
# GYM ENV  (imports gym lazily)
# ----------------------------------------------------------------------------
def make_env(waf=WAF, ep_len=EP_LEN, total_timesteps=TOTAL_TIMESTEPS):
    import gymnasium as gym
    import numpy as np
    from gymnasium import spaces

    obs_dim = 24 + (768 if USE_BERT else 0)

    class WafEvasionEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.action_space = spaces.Discrete(N_ACTIONS)
            self.observation_space = spaces.Box(low=-5.0, high=5.0,
                                                shape=(obs_dim,), dtype=np.float32)
            self.waf = waf
            self.ep_len = ep_len
            self.reward_fn = RewardCombiner(total_timesteps)
            self.checklist = SoftChecklist()
            self.bert = BertEncoder()
            self.fa2rnn = None   # set to a trained fa2rnn.FA2RNNReward to turn on R_REG
            self._step = 0

        def _obs(self, tele, action):
            ema = self.checklist.update(tele)
            eng = engineered_state(tele, action, self.waf, ema)
            vec = eng + self.bert.encode(tele, action, self.waf)
            return np.asarray(vec, dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._step = 0
            self.checklist = SoftChecklist()
            zero = {"found": False, "block_ratio": 0.0, "requests": 0,
                    "duration_s": 0.0, "techniques": [], "payloads": [],
                    "dbms": None, "waf_warned": False, "not_injectable": True,
                    "timed_out": False}
            noop = decode_action(0)
            return self._obs(zero, noop), {}

        def step(self, action_idx):
            action = decode_action(action_idx)
            tele = call_kali(action, waf=self.waf)
            reward, parts = self.reward_fn.compute(tele, fa2rnn=self.fa2rnn)
            self._step += 1
            terminated = bool(tele.get("found"))   # episode ends on a successful evasion
            truncated = self._step >= self.ep_len
            if DEBUG_EPISODES:
                print(f"[ep step={self._step}] {self.waf} {action['technique']}"
                      f"/L{action['level']}/R{action['risk']}/ra={action['random_agent']} "
                      f"found={tele.get('found')} block={tele.get('block_ratio')} "
                      f"req={tele.get('requests')} r={reward:.2f}"
                      + (f" ERR={tele['error']}" if tele.get('error') else ""), flush=True)
            info = {"telemetry": tele, "reward_parts": parts, "action": action}
            return self._obs(tele, action), float(reward), terminated, truncated, info

    return WafEvasionEnv()

# ----------------------------------------------------------------------------
# TRAIN  (DQN/PPO via stable-baselines3, with checkpoint/resume)
# ----------------------------------------------------------------------------
def train():
    import glob
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

    from circuit_breakers import CircuitBreaker

    cb = CircuitBreaker()   # shared by both callbacks (gold-eval feeds its gamma)

    class CircuitBreakerCallback(BaseCallback):
        """Dual breakers: lowers surrogate beta when gamma (reward-hacking) trips,
        halts the run on compute-runaway."""
        def __init__(self, env, breaker, verbose=0):
            super().__init__(verbose)
            self.env = env
            self.cb = breaker

        def _on_step(self):
            infos = self.locals.get("infos") or []
            rewards = self.locals.get("rewards")
            if infos and rewards is not None:
                tele = infos[0].get("telemetry", {})
                acts = self.cb.update(float(rewards[0]), tele, self.env.reward_fn)
                if acts.get("reduced_surrogate_beta") or acts.get("halt"):
                    print(f"[circuit-breaker] {acts}")
                if self.cb.tripped_runaway:
                    print("[circuit-breaker] compute runaway -> stopping training")
                    return False
            return True

    class GoldEvalCallback(BaseCallback):
        """C.6 gold-eval: every GOLD_EVAL_EVERY steps, freeze the policy and take
        GOLD_EVAL_K deterministic (eps=0) attempts on a separate env, recording the
        TRUE success rate (R_SYS only). Logs (step, proxy_reward, tpr_gold) to
        eval_log.json and feeds the breaker so gamma = proxy-vs-gold divergence."""
        def __init__(self, eval_env, breaker, every, k, verbose=0):
            super().__init__(verbose)
            self.eval_env = eval_env
            self.cb = breaker
            self.every = every
            self.k = k
            self._last = 0
            self.log = []

        def _on_step(self):
            if self.num_timesteps - self._last < self.every:
                return True
            self._last = self.num_timesteps
            found = 0
            for _ in range(self.k):
                obs, _ = self.eval_env.reset()
                a, _ = self.model.predict(obs, deterministic=True)
                _, _, _, _, info = self.eval_env.step(int(a))
                if info.get("telemetry", {}).get("found"):
                    found += 1
            tpr = found / max(1, self.k)
            buf = getattr(self.model, "ep_info_buffer", None)
            proxy = (sum(e["r"] for e in buf) / len(buf)) if buf else 0.0
            self.log.append({"episode": int(self.num_timesteps),
                             "proxy_reward": float(proxy), "tpr_gold": tpr})
            self.cb.feed_gold(proxy, tpr)
            try:
                with open(os.path.join(CKPT_DIR, "eval_log.json"), "w") as fh:
                    json.dump(self.log, fh)
            except Exception:
                pass
            print(f"[gold-eval] step={self.num_timesteps} proxy={proxy:.2f} "
                  f"TPR_gold={tpr:.3f} gamma={self.cb.gamma():.3f}")
            return True

    preflight()   # fail fast + clearly if the relay/tunnel is unreachable
    os.makedirs(CKPT_DIR, exist_ok=True)
    env = make_env()
    Algo = {"DQN": DQN, "PPO": PPO}[ALGO]

    resume = sorted(glob.glob(os.path.join(CKPT_DIR, "*.zip")))
    if resume:
        print(f"[resume] loading {resume[-1]}")
        model = Algo.load(resume[-1], env=env)
    else:
        kw = dict(policy="MlpPolicy", env=env, gamma=GAMMA, verbose=1,
                  tensorboard_log=os.path.join(CKPT_DIR, "tb"))
        if ALGO == "DQN":
            kw.update(buffer_size=20000, learning_starts=LEARNING_STARTS, train_freq=4,
                      target_update_interval=500, exploration_fraction=0.3)
        model = Algo(**kw)

    ckpt = CheckpointCallback(save_freq=500, save_path=CKPT_DIR,
                              name_prefix=f"{ALGO.lower()}_{WAF}")
    callbacks = [
        ckpt,
        CircuitBreakerCallback(env, cb),
        GoldEvalCallback(make_env(waf=WAF), cb, GOLD_EVAL_EVERY, GOLD_EVAL_K),
    ]
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks,
                log_interval=1, reset_num_timesteps=not bool(resume))
    model.save(os.path.join(CKPT_DIR, f"{ALGO.lower()}_{WAF}_final"))
    print("[done] saved final model")

# ----------------------------------------------------------------------------
# SMOKE TEST (no torch/gym) — validates relay + decode + reward end to end
# ----------------------------------------------------------------------------
def smoke(waf="luawaf"):
    rc = RewardCombiner(total_timesteps=10)
    cl = SoftChecklist()
    for idx in (0, 755):                       # first & last action
        a = decode_action(idx)
        # force random_agent on for luawaf so it isn't UA-blocked outright
        if waf == "luawaf":
            a["random_agent"] = True
        tele = call_kali(a, waf=waf, timeout=120)
        r, parts = rc.compute(tele)
        ema = cl.update(tele)
        print(f"idx={idx} action={a['technique']}/L{a['level']}/R{a['risk']}/"
              f"{a['tamper']}/ra={a['random_agent']} -> found={tele.get('found')} "
              f"block_ratio={tele.get('block_ratio')} reward={r:.2f} beta={parts['beta']:.2f}")
    print("checklist EMA:", [round(x, 3) for x in ema])


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        smoke(sys.argv[2] if len(sys.argv) > 2 else "luawaf")
    else:
        train()
