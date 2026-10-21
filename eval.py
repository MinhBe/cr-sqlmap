#!/usr/bin/env python3
"""
eval.py - CR-sqlmap evaluation suite (Phase 7), aligned to the master doc
(CR-sqlmap_benchmark_tong_hop.md sections, metric Groups A-D) and the reference
scripts in CR-sqlmap_C_chi_tiet.md (C.7). Isolated lab only.

  Group A - detection quality (shared with Paper 1 / Checklist-RL)
      TPR, FPR, Precision, F1, Request/Vuln, Request/Safe, Runtime
  Group B - evasion effectiveness + transferability (Paper 2 / Ouroboros, extended)
      TSR, FNR; cross-WAF transfer matrix TransferRate(A->B); Spearman rho (APCT vs rule depth)
  Group C - payload quality, PER TRAINING PHASE (early/mid/late)
      Duplicate Rate (MD5), TTR = V/N, Shannon Entropy (base-2) over tokens
  Group D - CR-sqlmap's own monitors (internal; for the circuit-breaker ablation)
      gamma = normalized area between proxy (R_SYN) and gold (TPR) curves  [A.6.1]
      FA2RNN covariate shift (acc drop vs baseline 0.8487)                 [A.6.2]
      rho_e = (TPR_e - TPR_baseline) / compute_hours_e  (cost-efficiency)   [A.7]

Multi-seed: seeds 42,123,456,789,1024; report mean +/- 95% CI.

All metric math is dependency-free (Spearman implemented here, not scipy) and
self-tested at the bottom. Rollout/model-load imports torch/SB3/gym lazily (Kaggle).
"""

import glob
import hashlib
import json
import math
import os
from collections import Counter

# ---------------------------------------------------------------------------
# generic stats
# ---------------------------------------------------------------------------
def mean_ci(values, conf=0.95):
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    m = sum(values) / n
    if n == 1:
        return (m, 0.0)
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    sem = math.sqrt(var) / math.sqrt(n)
    return (m, (1.96 if conf == 0.95 else 2.576) * sem)


def _rankdata(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0            # average rank, 1-based, ties shared
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman_rho(x, y):
    """Spearman rank correlation (no scipy)."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    rx, ry = _rankdata(x), _rankdata(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return cov / (vx * vy) if vx and vy else 0.0


# ---------------------------------------------------------------------------
# Group A - detection quality
# ---------------------------------------------------------------------------
def group_A(episodes):
    attempts = len(episodes)
    succ = [e for e in episodes if e.get("found")]
    tp = len(succ)
    fn = attempts - tp                                   # exploitable-but-missed
    tpr = tp / attempts if attempts else 0.0
    # "safe" inputs: episodes whose probe sent only the benign/original payload
    safe = [e for e in episodes if e.get("is_safe")]
    fp = sum(1 for e in safe if e.get("found"))
    tn = len(safe) - fp
    fpr = fp / len(safe) if safe else 0.0
    prec = tp / (tp + fp) if (tp + fp) else (1.0 if tp else 0.0)
    f1 = 2 * prec * tpr / (prec + tpr) if (prec + tpr) else 0.0
    return {
        "TPR": round(tpr, 4), "FPR": round(fpr, 4),
        "Precision": round(prec, 4), "F1": round(f1, 4),
        "Request_per_Vuln": round(sum(e.get("requests", 0) for e in succ) / tp, 2) if tp else None,
        "Request_per_Safe": round(sum(e.get("requests", 0) for e in safe) / len(safe), 2) if safe else None,
        "Runtime_s_mean": round(sum(e.get("duration_s", 0.0) for e in episodes) / attempts, 2) if attempts else 0.0,
        "attempts": attempts,
    }


# ---------------------------------------------------------------------------
# Group B - evasion effectiveness + transferability
# ---------------------------------------------------------------------------
def group_B(episodes, bypass_sets=None, apct_depth_pairs=None):
    attempts = len(episodes)
    tsr = sum(1 for e in episodes if e.get("found")) / attempts if attempts else 0.0
    out = {"TSR": round(tsr, 4), "FNR": round(1.0 - tsr, 4)}
    # cross-WAF transfer matrix: bypass_sets = {waf: set(payload_ids that bypass it)}
    if bypass_sets:
        wafs = list(bypass_sets)
        mat = {}
        for a in wafs:
            for b in wafs:
                if a == b:
                    continue
                A = bypass_sets[a]
                inter = len(A & bypass_sets[b])
                mat[f"{a}->{b}"] = round(inter / len(A), 4) if A else None
        out["transfer_matrix"] = mat
    # Spearman rho between APCT and rule-penetration depth (Ouroboros ref ~0.957)
    if apct_depth_pairs:
        ap = [p[0] for p in apct_depth_pairs]
        dp = [p[1] for p in apct_depth_pairs]
        out["spearman_APCT_depth"] = round(spearman_rho(ap, dp), 4)
    return out


# ---------------------------------------------------------------------------
# Group C - payload quality per phase  (doc C.7.2)
# ---------------------------------------------------------------------------
def _ttr(payloads):
    toks = [t for p in payloads for t in p.lower().split()]
    return (len(set(toks)) / max(1, len(toks)))


def _token_entropy(payloads):
    c = Counter(t for p in payloads for t in p.lower().split())
    N = sum(c.values())
    if not N:
        return 0.0
    return -sum((v / N) * math.log2(v / N) for v in c.values() if v)


def _dup_rate(payloads):
    h = [hashlib.md5(p.encode()).hexdigest() for p in payloads]
    return 1.0 - (len(set(h)) / max(1, len(h)))


def group_C(episodes, n_phases=3):
    n = len(episodes)
    if n == 0:
        return {}
    size = n // n_phases
    labels = ["early", "mid", "late"] if n_phases == 3 else [f"phase_{i+1}" for i in range(n_phases)]
    res = {}
    for i in range(n_phases):
        s = i * size
        e = (i + 1) * size if i < n_phases - 1 else n
        payloads = [ep.get("payload", "") for ep in episodes[s:e] if ep.get("payload")]
        res[labels[i]] = {
            "n_payloads": len(payloads),
            "Duplicate_Rate": round(_dup_rate(payloads), 4) if payloads else 0.0,
            "TTR": round(_ttr(payloads), 4) if payloads else 0.0,
            "Shannon_Entropy": round(_token_entropy(payloads), 4) if payloads else 0.0,
        }
    return res


# ---------------------------------------------------------------------------
# Group D - CR-sqlmap internal monitors
# ---------------------------------------------------------------------------
def gamma_overoptimization(proxy_series, gold_series):
    """gamma = normalized area where proxy (R_SYN) rises above gold (TPR) -
    the Goodhart divergence (A.6.1). Both series min-max normalized to [0,1] over
    aligned eval checkpoints; gamma in [0,1], ~0 = proxy tracks gold."""
    if not proxy_series or len(proxy_series) != len(gold_series):
        return 0.0

    def norm(s):
        lo, hi = min(s), max(s)
        return [(v - lo) / (hi - lo) if hi > lo else 0.0 for v in s]

    p, g = norm(proxy_series), norm(gold_series)
    return round(sum(max(0.0, pi - gi) for pi, gi in zip(p, g)) / len(p), 4)


def covariate_shift(fresh_acc, baseline_acc=0.8487):
    """FA2RNN accuracy drop on a fresh policy-sampled batch vs the baseline (A.6.2)."""
    return round(baseline_acc - fresh_acc, 4)


def rho_e(tpr_e, tpr_baseline, compute_hours_e):
    """Cost-efficiency slope (A.7): (TPR_e - TPR_baseline)/compute_hours_e."""
    return round((tpr_e - tpr_baseline) / compute_hours_e, 4) if compute_hours_e else 0.0


def group_D(eval_log=None, fresh_acc=None, baseline_tpr=0.0, checkpoints=None):
    """eval_log: [{episode, proxy_reward, tpr_gold}] from the held-out gold-eval
    loop (C.6). checkpoints: [{tpr, compute_hours}] for rho_e."""
    out = {}
    if eval_log:
        out["gamma"] = gamma_overoptimization([x["proxy_reward"] for x in eval_log],
                                              [x["tpr_gold"] for x in eval_log])
        # final gamma + first episode where proxy-gold gap opens materially
        out["final_gamma"] = out["gamma"]
    else:
        out["gamma"] = "needs gold-eval log (freeze policy every N eps, eval TPR on held-out set)"
    out["covariate_shift"] = covariate_shift(fresh_acc) if fresh_acc is not None \
        else "needs FA2RNN acc on a fresh policy-sampled batch"
    if checkpoints and len(checkpoints) >= 1:
        out["rho_e"] = [rho_e(c["tpr"], baseline_tpr, c["compute_hours"]) for c in checkpoints]
    else:
        out["rho_e"] = "needs per-checkpoint (tpr, compute_hours)"
    return out


# ---------------------------------------------------------------------------
# rollout (lazy torch/SB3/gym) - Kaggle
# ---------------------------------------------------------------------------
def rollout(checkpoint, n_episodes=50, seed=0, waf="modsec", deterministic=False):
    import random
    import numpy as np
    from stable_baselines3 import DQN, PPO
    import kaggle_waf_rl as K

    random.seed(seed); np.random.seed(seed)
    env = K.make_env(waf=waf)
    algo = {"DQN": DQN, "PPO": PPO}[os.environ.get("ALGO", "DQN")]
    model = algo.load(checkpoint, env=env)
    recs = []
    obs, _ = env.reset(seed=seed); done = 0
    while done < n_episodes:
        a, _ = model.predict(obs, deterministic=deterministic)
        obs, r, term, trunc, info = env.step(int(a))
        t = info.get("telemetry", {}); act = info.get("action", {})
        probe = t.get("probe", {}) or {}
        recs.append({
            "action_idx": int(a), "technique": act.get("technique"),
            "tamper": act.get("tamper", []),
            "payload": probe.get("mut", ""),            # Group C payload
            "found": bool(t.get("found")), "requests": t.get("requests", 0),
            "block_ratio": t.get("block_ratio", 0.0), "duration_s": t.get("duration_s", 0.0),
            "apct_mut": probe.get("apct_mut"), "reward": float(r),
            "beta": info.get("reward_parts", {}).get("beta"),
        })
        if term or trunc:
            obs, _ = env.reset(seed=seed + done + 1); done += 1
    return recs


def evaluate(ckpt_dir=None, seeds=(42, 123, 456, 789, 1024), n_episodes=50, waf="modsec"):
    ckpt_dir = ckpt_dir or os.environ.get("CKPT_DIR", "/kaggle/working/ckpt")
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.zip")))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints in {ckpt_dir}")
    report = {"waf": waf, "seeds": list(seeds), "n_episodes": n_episodes, "groupA_per_seed": []}
    runs = {s: rollout(ckpts[-1], n_episodes, s, waf) for s in seeds}
    for s in seeds:
        report["groupA_per_seed"].append(group_A(runs[s]))
        report.setdefault("groupB_per_seed", []).append(group_B(runs[s]))
    report["groupA_summary"] = {
        k: [round(x, 4) for x in mean_ci([a[k] for a in report["groupA_per_seed"] if a[k] is not None])]
        for k in ("TPR", "F1") }
    report["groupC_phases"] = group_C(runs[seeds[0]])
    # Group D gamma from the gold-eval loop's log (written by GoldEvalCallback)
    el_path = os.path.join(ckpt_dir, "eval_log.json")
    eval_log = json.load(open(el_path)) if os.path.exists(el_path) else None
    report["groupD"] = group_D(eval_log=eval_log)
    report["notes"] = ["Group B transfer_matrix/spearman need 2nd-WAF (:8081) bypass sets + "
                       "APCT/depth pairs; Group D covariate_shift needs a trained FA2RNN; "
                       "rho_e needs per-checkpoint compute_hours - wire from the run."]
    return report


# ---------------------------------------------------------------------------
# self-test (pure math)
# ---------------------------------------------------------------------------
def _selftest():
    ok = True
    # Group A
    eps = [{"found": True, "requests": 100, "duration_s": 5},
           {"found": True, "requests": 300, "duration_s": 7},
           {"found": False, "requests": 2000, "duration_s": 10},
           {"found": False, "requests": 5, "is_safe": True, "duration_s": 1}]
    a = group_A(eps)
    ok &= a["TPR"] == 0.5 and a["Request_per_Vuln"] == 200.0 and a["FPR"] == 0.0
    print("Group A:", a, "OK" if ok else "FAIL")
    # Spearman (monotone -> ~1)
    rho = spearman_rho([1, 2, 3, 4, 5], [2, 4, 5, 9, 11])
    ok &= rho > 0.98
    print("spearman monotone:", round(rho, 4), "OK")
    # Group B transfer matrix
    b = group_B([{"found": True}, {"found": False}],
                bypass_sets={"modsec": {1, 2, 3}, "luawaf": {2, 3, 4}},
                apct_depth_pairs=[(0.1, 1), (0.2, 2), (0.3, 3), (0.4, 4)])
    ok &= b["TSR"] == 0.5 and b["transfer_matrix"]["modsec->luawaf"] == round(2/3, 4) and b["spearman_APCT_depth"] == 1.0
    print("Group B:", b, "OK")
    # Group C: late phase duplicated -> high dup, low TTR/entropy
    c_eps = ([{"payload": f"union select {i}"} for i in range(9)] +
             [{"payload": "1 or 1=1"}] * 9)
    c = group_C(c_eps)
    ok &= c["late"]["Duplicate_Rate"] > c["early"]["Duplicate_Rate"]
    print("Group C early vs late dup:", c["early"]["Duplicate_Rate"], c["late"]["Duplicate_Rate"], "OK")
    # Group D gamma: proxy rising above flat gold -> gamma>0
    g = gamma_overoptimization([1, 2, 3, 4, 5], [1, 1, 1, 1, 1])
    g0 = gamma_overoptimization([1, 2, 3], [1, 2, 3])
    ok &= g > 0.3 and g0 == 0.0
    print(f"Group D gamma overopt={g} tracking={g0}", "OK")
    ok &= covariate_shift(0.60) == round(0.8487 - 0.60, 4) and rho_e(0.8, 0.5, 3.0) == 0.1
    print("Group D covariate/rho_e OK")
    print("\nEVAL METRIC MATH:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        print(json.dumps(evaluate(waf=os.environ.get("WAF", "modsec")), indent=2))
    else:
        _selftest()
