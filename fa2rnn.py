#!/usr/bin/env python3
"""
fa2rnn.py - FA2RNN for R_REG (Phase 6, Tier-4). CR-sqlmap RL lab.

Reimplementation of FA-RNN (Jiang et al., EMNLP 2020, "Cold-Start and
Interpretability: Turning Regular Expressions into Trainable Recurrent Neural
Networks"; ref RE2NN classification variant) adapted to SQLi-detection regexes.

Pipeline (Ouroboros spec):
    regex --> NFA (Thompson) --> DFA (subset constr.) --> minimize (Moore)
          --> weighted-FA 5-tuple (alpha, {M_s}, eta)
          --> CP/PARAFAC decomposition of the transition tensor  (tensorly)
          --> linear-activation RNN  h_t = ((h_{t-1} . D1) (*) v_t) . D2^T  (Eq4)
          --> MLP head --> psi  (malicious probability)

What is VALIDATED here (pure Python, no deps): the symbolic core
regex->NFA->DFA->minimize->weighted-FA, checked against Python `re`.
The tensor/torch parts (CP decomp, RNN, MLP, training) import numpy/tensorly/
torch lazily and are meant to run on Kaggle (GPU); they are NOT exercised in the
Remote-Control environment (no torch there).

R_REG wiring: FA2RNNReward.delta(orig, mut) = psi(orig) - psi(mut). It stays a
no-op (0.0) until the executor emits the original-vs-mutated payload PAIR per
step (same remaining-wire as R_RUL's timing baseline); the RewardCombiner hook
already blends it behind the annealed beta.

[OPEN - confirm vs master doc / ref repo before training]
  * MLP head depth/width/activation, CP rank R, loss/optimizer: defaulted to the
    paper (positives-only, 70/30, 10 epochs, lr 0.01, Adam; target ~F1 0.86).
  * Seed regexes below are canonical SQLi in the supported subset; distilling the
    full CRS-942 PCRE into this subset is the GA-rule-extraction follow-on.
"""

# ===========================================================================
# PART A - symbolic core (pure Python, validated here)
# ===========================================================================

EPS = None          # epsilon transition marker
ANY = "\x00ANY"     # internal marker for '.' (expanded over the alphabet)


class _NFA:
    def __init__(self):
        self.trans = {}      # state -> list[(symbol, state)]  (symbol EPS or char or ANY)
        self.start = None
        self.accept = None
        self._n = 0

    def new_state(self):
        s = self._n
        self._n += 1
        self.trans[s] = []
        return s

    def add(self, src, sym, dst):
        self.trans[src].append((sym, dst))


def _parse_to_nfa(pattern):
    """Recursive-descent over the supported subset: literals, '.', '*','+','?',
    '|', '(...)'. Returns a Thompson NFA with a single start and accept state."""
    pos = 0
    n = len(pattern)
    nfa = _NFA()

    def peek():
        return pattern[pos] if pos < n else None

    def eat():
        nonlocal pos
        c = pattern[pos]
        pos += 1
        return c

    def atom():
        c = peek()
        if c == "(":
            eat()
            s, a = expr()
            assert peek() == ")", "unbalanced ("
            eat()
            return s, a
        if c == "\\":          # escaped literal
            eat()
            lit = eat()
            s = nfa.new_state(); a = nfa.new_state(); nfa.add(s, lit, a)
            return s, a
        c = eat()
        sym = ANY if c == "." else c
        s = nfa.new_state(); a = nfa.new_state(); nfa.add(s, sym, a)
        return s, a

    def factor():
        s, a = atom()
        while peek() in ("*", "+", "?"):
            op = eat()
            ns, na = nfa.new_state(), nfa.new_state()
            if op == "*":
                nfa.add(ns, EPS, s); nfa.add(ns, EPS, na)
                nfa.add(a, EPS, s); nfa.add(a, EPS, na)
            elif op == "+":
                nfa.add(ns, EPS, s)
                nfa.add(a, EPS, s); nfa.add(a, EPS, na)
            else:  # '?'
                nfa.add(ns, EPS, s); nfa.add(ns, EPS, na); nfa.add(a, EPS, na)
            s, a = ns, na
        return s, a

    def term():
        # concatenation of factors
        s = a = None
        while peek() not in (None, "|", ")"):
            fs, fa = factor()
            if s is None:
                s, a = fs, fa
            else:
                nfa.add(a, EPS, fs); a = fa
        if s is None:          # empty -> epsilon
            s = nfa.new_state(); a = nfa.new_state(); nfa.add(s, EPS, a)
        return s, a

    def expr():
        s, a = term()
        if peek() == "|":
            branches = [(s, a)]
            while peek() == "|":
                eat(); branches.append(term())
            ns, na = nfa.new_state(), nfa.new_state()
            for bs, ba in branches:
                nfa.add(ns, EPS, bs); nfa.add(ba, EPS, na)
            s, a = ns, na
        return s, a

    s, a = expr()
    nfa.start, nfa.accept = s, a
    return nfa


def _eps_closure(nfa, states):
    stack = list(states); seen = set(states)
    while stack:
        st = stack.pop()
        for sym, dst in nfa.trans[st]:
            if sym is EPS and dst not in seen:
                seen.add(dst); stack.append(dst)
    return frozenset(seen)


class DFA:
    def __init__(self, start, accepts, delta, alphabet):
        self.start = start                # int
        self.accepts = accepts            # set[int]
        self.delta = delta                # dict[(int, sym)] -> int
        self.alphabet = alphabet          # list[str]

    def accepts_str(self, s):
        q = self.start
        for ch in s:
            key = (q, ch if ch in self.alphabet else None)
            if key not in self.delta:
                return False
            q = self.delta[key]
        return q in self.accepts


def nfa_to_dfa(nfa, alphabet):
    start = _eps_closure(nfa, {nfa.start})
    id_of = {start: 0}
    delta = {}
    work = [start]
    accepts = set()
    while work:
        cur = work.pop()
        ci = id_of[cur]
        if nfa.accept in cur:
            accepts.add(ci)
        for sym in alphabet:
            move = set()
            for st in cur:
                for s2, dst in nfa.trans[st]:
                    if s2 == sym or s2 == ANY:
                        move.add(dst)
            if not move:
                continue
            nxt = _eps_closure(nfa, move)
            if nxt not in id_of:
                id_of[nxt] = len(id_of); work.append(nxt)
            delta[(ci, sym)] = id_of[nxt]
    return DFA(0, accepts, delta, list(alphabet))


def minimize_dfa(dfa):
    """Moore partition-refinement minimization (total DFA via implicit dead state)."""
    states = set([dfa.start]) | set(dfa.accepts) | {v for v in dfa.delta.values()} \
        | {k[0] for k in dfa.delta}
    states = sorted(states)
    DEAD = -1
    sigma = dfa.alphabet

    def step(q, a):
        return dfa.delta.get((q, a), DEAD)

    # initial partition: accept / non-accept (+ dead)
    part = {}
    for q in states:
        part[q] = 0 if q in dfa.accepts else 1
    part[DEAD] = 1
    changed = True
    while changed:
        changed = False
        sig = {}
        for q in list(part):
            sig[q] = (part[q], tuple(part[step(q, a)] for a in sigma))
        groups = {}
        for q, s in sig.items():
            groups.setdefault(s, []).append(q)
        newpart = {}
        for i, (_, members) in enumerate(sorted(groups.items())):
            for q in members:
                newpart[q] = i
        if newpart != part:
            part = newpart; changed = True
    # rebuild
    new_start = part[dfa.start]
    new_delta = {}
    new_accepts = set()
    for q in states:
        if q in dfa.accepts:
            new_accepts.add(part[q])
        for a in sigma:
            t = step(q, a)
            if t != DEAD:
                new_delta[(part[q], a)] = part[t]
    return DFA(new_start, new_accepts, new_delta, list(sigma))


def build_dfa(regexes, alphabet):
    """Build one DFA over `alphabet` (+OOV). Use only for a SINGLE regex; unioning
    many '.*'-anchored rules here is worst-case exponential (see build_wfas)."""
    alphabet = list(alphabet) + ([OOV] if OOV not in alphabet else [])
    big = _NFA()
    s0 = big.new_state(); a0 = big.new_state()
    # build each sub-NFA in its own index space, then splice via offset
    for rx in regexes:
        sub = _parse_to_nfa(rx)
        off = big._n
        for st, edges in sub.trans.items():
            big.trans[st + off] = [(sym, d + off) for sym, d in edges]
        big._n = off + sub._n
        big.add(s0, EPS, sub.start + off)
        big.add(sub.accept + off, EPS, a0)
    big.start, big.accept = s0, a0
    return nfa_to_dfa(big, alphabet)


# ---- weighted FA 5-tuple (alpha, {M_s}, eta) over the DFA ------------------
def dfa_to_weighted_fa(dfa):
    states = sorted(set([dfa.start]) | dfa.accepts |
                    {v for v in dfa.delta.values()} | {k[0] for k in dfa.delta})
    idx = {q: i for i, q in enumerate(states)}
    nq = len(states)
    alpha = [0.0] * nq; alpha[idx[dfa.start]] = 1.0
    eta = [1.0 if q in dfa.accepts else 0.0 for q in states]
    M = {a: [[0.0] * nq for _ in range(nq)] for a in dfa.alphabet}
    for (q, a), t in dfa.delta.items():
        M[a][idx[q]][idx[t]] = 1.0
    return {"alpha": alpha, "eta": eta, "M": M, "states": nq,
            "alphabet": list(dfa.alphabet)}


OOV = "\x02"   # out-of-vocabulary symbol; '.' (ANY) expands over it too


def wfa_accepts(wfa, s):
    """Row-vector run: v = alpha; v <- v . M_s; accept iff v . eta > 0.
    Chars outside the alphabet map to OOV (matched by '.' / ANY, never literals)."""
    nq = wfa["states"]
    v = list(wfa["alpha"])
    for ch in s:
        sym = ch if ch in wfa["M"] else OOV
        M = wfa["M"][sym]
        v = [sum(v[i] * M[i][j] for i in range(nq)) for j in range(nq)]
    return sum(vi * ei for vi, ei in zip(v, wfa["eta"])) > 0


def build_single_wfa(rx, alphabet):
    """One regex -> minimized DFA -> weighted-FA (small, linear in |rx|)."""
    return dfa_to_weighted_fa(minimize_dfa(build_dfa([rx], alphabet)))


def build_wfas(regexes, alphabet):
    """Per-regex weighted FAs (NEVER union into one DFA: that is worst-case
    exponential for '.*'-anchored substring rules)."""
    return [build_single_wfa(rx, alphabet) for rx in regexes]


def classify_any(wfas, s):
    """Malicious iff ANY rule's FA accepts (decision-level union)."""
    return any(wfa_accepts(w, s) for w in wfas)


# ===========================================================================
# PART B - seed regexes (supported subset; canonical SQLi, CRS-942 inspired)
# ===========================================================================
# Lowercased; '.' = any alphabet symbol. Distilling full CRS-942 PCRE into this
# subset is the GA-rule-extraction follow-on.
SEED_SQLI_REGEXES = [
    ".*union.*select.*",
    ".*or.*=.*",
    ".*and.*=.*",
    ".*sleep\\(.*",
    ".*benchmark\\(.*",
    ".*information_schema.*",
    ".*concat\\(.*",
    ".*'.*or.*'.*",
    ".*--.*",
    ".*/\\*.*\\*/.*",
]
DEFAULT_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789 _=()'\"-/*,.;%@#"


# ===========================================================================
# PART C - neural layers (torch/tensorly, run on Kaggle; lazy imports)
# ===========================================================================
def cp_decompose(wfas, rank=20):
    """Build the BLOCK-DIAGONAL transition tensor T[|Sigma|, N, N] over the direct
    sum of the per-regex automata (N = sum of states; linear, never the product),
    then CP/PARAFAC-decompose into rank-R factors (A: symbol, D1: from, D2: to)."""
    import numpy as np
    import tensorly as tl
    from tensorly.decomposition import parafac
    sigma = wfas[0]["alphabet"]
    N = sum(w["states"] for w in wfas)
    T = np.zeros((len(sigma), N, N), dtype=np.float32)
    alpha = np.zeros(N, np.float32); eta = np.zeros(N, np.float32)
    off = 0
    for w in wfas:
        nq = w["states"]
        for i in range(nq):
            alpha[off + i] = w["alpha"][i]; eta[off + i] = w["eta"][i]
        for si, a in enumerate(sigma):
            T[si, off:off + nq, off:off + nq] = np.asarray(w["M"][a], np.float32)
        off += nq
    weights, factors = parafac(tl.tensor(T), rank=rank, init="random", n_iter_max=200)
    A, D1, D2 = [np.asarray(f, dtype=np.float32) for f in factors]
    return {"A": A, "D1": D1, "D2": D2, "rank": rank, "states": N,
            "alphabet": list(sigma), "alpha": alpha, "eta": eta}


def build_fa2rnn(regexes=None, alphabet=DEFAULT_ALPHABET, rank=20, mlp_hidden=64):
    """Assemble the trainable FA2RNN model from seed regexes. torch required."""
    import numpy as np
    import torch
    import torch.nn as nn

    regexes = regexes or SEED_SQLI_REGEXES
    wfas = build_wfas(regexes, alphabet)        # per-regex (linear), not unioned
    cp = cp_decompose(wfas, rank=rank)          # block-diagonal tensor -> CP

    class FA2RNN(nn.Module):
        """Linear-activation RNN (Eq4) initialised from the CP factors + MLP head.
            h_t = ((h_{t-1} @ D1) * v_t) @ D2^T ,  v_t = A[char]
        then psi = sigmoid(MLP(h_T))."""
        def __init__(self):
            super().__init__()
            self.alphabet = {c: i for i, c in enumerate(cp["alphabet"])}
            self.rank = cp["rank"]
            self.A = nn.Parameter(torch.tensor(cp["A"]))     # |Sigma| x R
            self.D1 = nn.Parameter(torch.tensor(cp["D1"]))   # nq x R
            self.D2 = nn.Parameter(torch.tensor(cp["D2"]))   # nq x R
            self.alpha = torch.tensor(cp["alpha"])           # nq
            self.eta = torch.tensor(cp["eta"])               # nq
            self.mlp = nn.Sequential(
                nn.Linear(cp["D1"].shape[0] + 1, mlp_hidden), nn.ReLU(),
                nn.Linear(mlp_hidden, 1))

        def _run(self, s):
            h = self.alpha.clone()                           # nq
            for ch in s.lower():
                si = self.alphabet.get(ch)
                if si is None:
                    continue
                v = self.A[si]                               # R
                h = ((h @ self.D1) * v) @ self.D2.t()        # Eq4 -> nq
            score = torch.dot(h, self.eta).unsqueeze(0)      # FA accept score
            feat = torch.cat([h, score])
            return torch.sigmoid(self.mlp(feat)).squeeze()   # psi in (0,1)

        def forward(self, strings):
            return torch.stack([self._run(s) for s in strings])

    return FA2RNN()


def train_fa2rnn(model, positives, epochs=10, lr=0.01, split=0.7):
    """Positives-only training (paper setup). Returns the trained model."""
    import torch
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = torch.nn.BCELoss()
    k = int(len(positives) * split)
    train, val = positives[:k], positives[k:]
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        pred = model(train)
        loss = bce(pred, torch.ones_like(pred))
        loss.backward(); opt.step()
        with torch.no_grad():
            vp = model(val) if val else pred
            vacc = float((vp > 0.5).float().mean())
        print(f"[fa2rnn] epoch {ep+1}/{epochs} loss={loss.item():.4f} val_acc={vacc:.3f}")
    return model


class FA2RNNReward:
    """R_REG provider: delta = psi(orig) - psi(mut). Returns 0.0 until a trained
    model AND an orig/mut payload pair are supplied (same remaining wire as R_RUL)."""
    def __init__(self, model=None):
        self.model = model

    def psi(self, s):
        if self.model is None:
            return 0.0
        import torch
        with torch.no_grad():
            return float(self.model([s])[0])

    def delta(self, orig=None, mut=None):
        if self.model is None or orig is None or mut is None:
            return 0.0
        return self.psi(orig) - self.psi(mut)


# ===========================================================================
# self-test - symbolic core only (no torch)
# ===========================================================================
if __name__ == "__main__":
    import re as _re

    alpha = "abcd"
    cases = [
        ("a(b|c)*d", ["ad", "abd", "acd", "abbccd", "abcbcd"], ["a", "abc", "d", "abx"]),
        ("ab*c",     ["ac", "abc", "abbbc"],                   ["abce", "ab", "c"]),
        ("(a|b)+",   ["a", "b", "abab", "bbbb"],               ["", "c", "aac?"]),
    ]
    ok = True
    for rx, yes, no in cases:
        wfa = build_single_wfa(rx, alpha)
        dfa = minimize_dfa(build_dfa([rx], alpha))
        for s in yes:
            d, w = dfa.accepts_str(s), wfa_accepts(wfa, s)
            ref = bool(_re.fullmatch(rx, s))
            print(f"  /{rx}/  accept '{s}': dfa={d} wfa={w} re={ref}")
            ok &= d and w and ref
        for s in no:
            d, w = dfa.accepts_str(s), wfa_accepts(wfa, s)
            print(f"  /{rx}/  reject '{s}': dfa={d} wfa={w}")
            ok &= (not d) and (not w)

    # SQLi seed set: per-regex automata, decision-level union (fast; no blow-up)
    print("\n[seed SQLi - per-regex automata]")
    wfas = build_wfas(SEED_SQLI_REGEXES, DEFAULT_ALPHABET)
    print(f"  rules={len(wfas)} states(sum)={sum(w['states'] for w in wfas)} "
          f"|alphabet|={len(wfas[0]['alphabet'])}")
    for s, exp in [("1 union select pass", True), ("1' or '1'='1", True),
                   ("sleep(5)", True), ("select concat(a)", True),
                   ("hello world", False), ("get the id now", False)]:
        acc = classify_any(wfas, s.lower())
        print(f"  '{s}' -> malicious={acc} (expect {exp})  {'OK' if acc==exp else 'CHECK'}")
        ok &= (acc == exp)
    print("\nSYMBOLIC CORE:", "PASS" if ok else "FAIL")
