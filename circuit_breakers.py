#!/usr/bin/env python3
"""
circuit_breakers.py - dual circuit breakers for CR-sqlmap RL training (Phase 6),
aligned to the master doc (A.6 reward-hacking, A.7 cost).

1) Anti-reward-hacking (A.6.1): gamma = normalized area where the proxy reward
   (R_SYN) rises ABOVE the gold reward (true TPR from the held-out eval loop).
   gamma ~ 0 means the proxy tracks gold; large gamma = Goodhart divergence.
   When gamma > threshold we REDUCE the surrogate weight beta (anneal it down
   faster) so the true outcome dominates - i.e. reduce (1-beta_e)->... in the
   doc's convention, which in this code (R=(1-beta)*R_outcome + beta*R_SYN) means
   lowering beta. Feed gold via feed_gold(); until a gold-eval loop runs, an
   online heuristic gamma (reward earned on non-success episodes) is the early
   proxy.

2) Anti-compute-runaway (A.7): hard caps on cumulative sqlmap requests and
   wall-clock; tripping halts training so an unattended run can't overrun.

Pure Python (no torch/gym) - unit-testable; the notebook wraps it as an SB3
BaseCallback that calls update() each step (and feed_gold() each gold-eval).
"""

import time
from collections import deque


def gamma_from_curves(proxy_series, gold_series):
    """gamma = mean over aligned checkpoints of max(0, proxy_norm - gold_norm),
    each series min-max normalized to [0,1]. Range [0,1]. (A.6.1)"""
    if not proxy_series or len(proxy_series) != len(gold_series):
        return 0.0

    def _n(s):
        lo, hi = min(s), max(s)
        return [(v - lo) / (hi - lo) if hi > lo else 0.0 for v in s]

    p, g = _n(proxy_series), _n(gold_series)
    return sum(max(0.0, a - b) for a, b in zip(p, g)) / len(p)


class CircuitBreaker:
    def __init__(self, window=50, gamma_threshold=0.15, beta_step=0.05,
                 beta_floor=0.0, max_requests=2_000_000, max_wall_s=26 * 3600):
        self.window = window
        self.gamma_threshold = gamma_threshold
        self.beta_step = beta_step
        self.beta_floor = beta_floor
        self.max_requests = max_requests
        self.max_wall_s = max_wall_s

        self._hist = deque(maxlen=window)       # (r_total, found) - online heuristic
        self._gold = []                          # [(proxy_R_SYN, gold_TPR)] from eval loop
        self.total_requests = 0
        self._t0 = time.time()
        self.trips = {"reward_hacking": 0, "compute_runaway": 0}
        self.tripped_runaway = False

    # authoritative gamma, once gold-eval samples are available
    def feed_gold(self, proxy_rsyn, gold_tpr):
        self._gold.append((float(proxy_rsyn), float(gold_tpr)))

    def gamma(self):
        if len(self._gold) >= 2:
            return gamma_from_curves([p for p, _ in self._gold],
                                     [g for _, g in self._gold])
        # online early-proxy: share of reward on episodes without a real success
        if not self._hist:
            return 0.0
        tot = sum(r for r, _ in self._hist)
        if tot <= 0:
            return 0.0
        return sum(r for r, f in self._hist if not f) / tot

    def update(self, r_total, telemetry, reward_combiner, now=None):
        self._hist.append((float(r_total), bool(telemetry.get("found"))))
        self.total_requests += int(telemetry.get("requests", 0) or 0)
        actions = {}
        g = self.gamma()
        actions["gamma"] = round(g, 3)

        ready = len(self._gold) >= 2 or len(self._hist) >= self.window
        if ready and g > self.gamma_threshold:
            rc = reward_combiner
            rc.b1 = max(self.beta_floor, rc.b1 - self.beta_step)   # drop the floor
            rc.b0 = max(rc.b1, rc.b0 - self.beta_step)             # and the start
            actions["reduced_surrogate_beta"] = {"b0": round(rc.b0, 3), "b1": round(rc.b1, 3)}
            self.trips["reward_hacking"] += 1

        wall = (now if now is not None else time.time()) - self._t0
        if self.total_requests > self.max_requests or wall > self.max_wall_s:
            self.tripped_runaway = True
            self.trips["compute_runaway"] += 1
            actions["halt"] = {"requests": self.total_requests, "wall_s": round(wall)}
        return actions


# --- self-test -------------------------------------------------------------
if __name__ == "__main__":
    class _RC:
        b0 = 0.90
        b1 = 0.15

    # 1) reward-hacking (online heuristic): reward with NO success -> gamma high
    #    -> surrogate beta is pushed DOWN (b0,b1 fall)
    rc = _RC(); cb = CircuitBreaker(window=50)
    for _ in range(60):
        cb.update(5.0, {"found": False, "requests": 10}, rc)
    print(f"[reward-hacking] gamma={cb.gamma():.2f} b0={rc.b0:.2f} b1={rc.b1:.2f} "
          f"-> {'PASS' if rc.b1 < 0.15 and rc.b0 < 0.90 else 'FAIL'}")

    # 2) authoritative gamma from proxy-gold curves: proxy rises, gold flat
    g = gamma_from_curves([1, 2, 3, 4, 5], [1, 1, 1, 1, 1])
    g0 = gamma_from_curves([1, 2, 3], [1, 2, 3])
    print(f"[gamma curves] overopt={g:.2f} tracking={g0:.2f} "
          f"-> {'PASS' if g > 0.3 and g0 == 0.0 else 'FAIL'}")

    # 3) healthy: all reward from real successes -> gamma 0 -> beta untouched
    rc2 = _RC(); cb2 = CircuitBreaker(window=50)
    for _ in range(60):
        cb2.update(10.0, {"found": True, "requests": 300}, rc2)
    print(f"[healthy] gamma={cb2.gamma():.2f} b0={rc2.b0:.2f} "
          f"-> {'PASS' if rc2.b0 == 0.90 else 'FAIL'}")

    # 4) compute runaway
    rc3 = _RC(); cb3 = CircuitBreaker(max_requests=1000)
    cb3.update(0.0, {"found": False, "requests": 5000}, rc3)
    print(f"[runaway] tripped={cb3.tripped_runaway} -> {'PASS' if cb3.tripped_runaway else 'FAIL'}")

    rc4 = _RC(); cb4 = CircuitBreaker(max_wall_s=10)
    cb4.update(0.0, {"found": True, "requests": 1}, rc4, now=cb4._t0 + 20)
    print(f"[wallclock] tripped={cb4.tripped_runaway} -> {'PASS' if cb4.tripped_runaway else 'FAIL'}")
