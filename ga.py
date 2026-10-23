#!/usr/bin/env python3
"""
ga.py - Ouroboros GA rule-extraction (Phase 4), faithful to the locked math spec
(CR-sqlmap_Ouroboros_math_spec.md section 1).

Blocked payloads are used to REVERSE-ENGINEER the WAF rule as a generalized regex, which
FA2RNN (fa2rnn.py) then turns into the dense reward R_REG. Pipeline (spec 1.1-1.4):

  blocked payloads
    -> endpoint-erosion (strip non-triggering common prefix/suffix)        [PIN]
    -> TF-IDF (<=1000 feats) + DBSCAN(eps=4.3, min_samples=5) clustering     [PIN]
    -> per cluster: LCS-align members into columns                          [PIN]
    -> GA over 16-gene hex genotypes (one gene per column) minimising
         L(r) = a*FN + b*FP + g_len*len(r)/Lmax + g_c*Cplx(r)               [DEC weights]
       pop=100, gen=100, mut=5%, cross=30%, tournament k=3, elitism=2       [PIN/DEC]
    -> best regex per cluster  (approximation of the WAF rule family)

GUIDED variant (Minh's contribution, declared, NOT faithful-core): the input corpus can be
*seeded/directed* from sqlmap-confirmed evasive payloads instead of only raw blocked ones.
That changes WHICH payloads the GA generalises, never the GA math. See run_ga(..., guided=).

Deps: scikit-learn (TfidfVectorizer, DBSCAN), numpy. Pure-python otherwise.
"""
import os
import random
import re

# --- 16-gene alphabet (spec Table 1) ----------------------------------------
# Fixed-token genes 0x0..0xb. Data-dependent genes 0xc..0xf decode from the aligned
# column's real characters.
_FIXED = {
    0x0: r"\d", 0x1: r"[A-Z]", 0x2: r"[a-z]", 0x3: r"[A-Za-z]",
    0x4: r"[0-9A-F]", 0x5: r"[0-9a-f]", 0x6: r"\w", 0x7: r"\s",
    0x8: r"[ ]", 0x9: r".", 0xa: r"[{}^$.|*+?]",
    0xb: r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]",
}
# Complexity weight per gene (spec 1.3): literal .1 / narrow class .3 / \w,\s .6 / dot 1.0.
# char-or/string-or (0xe/0xf) are the most specific -> treated as literal-ish (.1).
_CPLX = {0x9: 1.0, 0x6: 0.6, 0x7: 0.6, 0xe: 0.1, 0xf: 0.1}
GENE_MAX = 0x10  # 16 symbols

ALPHA = float(os.environ.get("GA_ALPHA", "1.0"))
BETA = float(os.environ.get("GA_BETA", "1.0"))
GAMMA_LEN = float(os.environ.get("GA_GAMMA_LEN", "0.05"))
GAMMA_C = float(os.environ.get("GA_GAMMA_C", "0.2"))
LMAX = float(os.environ.get("GA_LMAX", "100.0"))
POP = int(os.environ.get("GA_POP", "100"))
GENERATIONS = int(os.environ.get("GA_GEN", "100"))
P_MUT = float(os.environ.get("GA_MUT", "0.05"))
P_CROSS = float(os.environ.get("GA_CROSS", "0.30"))
TOURNAMENT_K = int(os.environ.get("GA_TOURNEY", "3"))
ELITISM = int(os.environ.get("GA_ELITISM", "2"))

DEFAULT_NEGATIVES = ["1", "2", "100", "admin", "name", "gordonb", "test@x.com",
                     "hello world", "/page.php", "SELECT_this_is_text"]


# --- 1.1 preprocessing -------------------------------------------------------
def endpoint_erosion(payloads):
    """Strip the common prefix/suffix shared by all payloads (non-triggering boundaries),
    keeping the rule-triggering core."""
    if not payloads:
        return payloads
    pre = os.path.commonprefix(payloads)
    suf = os.path.commonprefix([p[::-1] for p in payloads])[::-1]
    out = []
    for p in payloads:
        core = p[len(pre):len(p) - len(suf)] if (len(pre) + len(suf) < len(p)) else p
        out.append(core or p)
    return out


def cluster(payloads):
    """TF-IDF(<=1000 char n-gram feats) + DBSCAN(eps=4.3, min_samples=5). Returns
    {label: [payloads]} (label -1 = noise, dropped). Needs scikit-learn."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import DBSCAN
    if len(payloads) < 2:
        return {0: list(payloads)} if payloads else {}
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), max_features=1000)
    X = vec.fit_transform(payloads).toarray()
    labels = DBSCAN(eps=4.3, min_samples=5, metric="euclidean").fit_predict(X)
    groups = {}
    for p, lab in zip(payloads, labels):
        if lab == -1:
            continue
        groups.setdefault(int(lab), []).append(p)
    if not groups:  # everything noise -> treat all as one family so GA still runs
        groups = {0: list(payloads)}
    return groups


# --- 1.2 LCS alignment -> columns -------------------------------------------
def _lcs_map(ref, s):
    """LCS DP of ref vs s; return dict {ref_index: char_from_s} for aligned positions."""
    n, m = len(ref), len(s)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            dp[i][j] = dp[i + 1][j + 1] + 1 if ref[i] == s[j] else max(dp[i + 1][j], dp[i][j + 1])
    out, i, j = {}, 0, 0
    while i < n and j < m:
        if ref[i] == s[j]:
            out[i] = s[j]; i += 1; j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return out


def align_columns(members):
    """Align cluster members by LCS against the longest member. Returns a list of columns;
    each column = {'chars': set, 'optional': bool} over the reference positions."""
    ref = max(members, key=len)
    cols = [{"chars": {ref[i]}, "hits": 1} for i in range(len(ref))]
    for s in members:
        if s is ref:
            continue
        mp = _lcs_map(ref, s)
        for i in range(len(ref)):
            if i in mp:
                cols[i]["chars"].add(mp[i]); cols[i]["hits"] += 1
    total = len(members)
    return [{"chars": c["chars"], "optional": c["hits"] < total} for c in cols]


# --- 1.2 genotype decode -----------------------------------------------------
def _decode_gene(gene, chars):
    """Map one gene to a regex token, using column `chars` for data-dependent genes."""
    if gene in _FIXED:
        return _FIXED[gene]
    codes = sorted(ord(c) for c in chars)
    if gene == 0xc:  # range-for-all
        return f"[{re.escape(chr(codes[0]))}-{re.escape(chr(codes[-1]))}]"
    if gene == 0xd:  # range-for-letters
        letters = sorted(c for c in chars if c.isalpha())
        if letters:
            return f"[{re.escape(letters[0])}-{re.escape(letters[-1])}]"
        return r"[A-Za-z]"
    if gene == 0xe:  # char-or (distinct chars)
        return "[" + "".join(re.escape(c) for c in sorted(chars)) + "]"
    # 0xf string-or: single-char columns -> char class; keep it a class here
    return "[" + "".join(re.escape(c) for c in sorted(chars)) + "]"


def genotype_to_regex(genotype, columns):
    parts = []
    for gene, col in zip(genotype, columns):
        tok = _decode_gene(gene, col["chars"])
        parts.append(tok + ("?" if col["optional"] else ""))
    return "".join(parts)


def _cplx(genotype):
    return sum(_CPLX.get(g, 0.3) for g in genotype)


# --- 1.3 fitness -------------------------------------------------------------
def fitness(genotype, columns, members, negatives):
    """L(r) = a*FN + b*FP + g_len*len/Lmax + g_c*Cplx;  returns (fitness=-L, regex)."""
    rx = genotype_to_regex(genotype, columns)
    try:
        pat = re.compile(rx)
    except re.error:
        return -1e9, rx  # invalid regex -> lethal
    fn = sum(1 for m in members if not pat.search(m))
    fp = sum(1 for n in negatives if pat.search(n))
    L = (ALPHA * fn + BETA * fp
         + GAMMA_LEN * len(rx) / LMAX + GAMMA_C * _cplx(genotype))
    return -L, rx


# --- 1.4 evolution -----------------------------------------------------------
def _tournament(scored):
    return max(random.sample(scored, min(TOURNAMENT_K, len(scored))), key=lambda x: x[0])[1]


def evolve(columns, members, negatives, rng=None):
    """Run the GA over genotypes (len = #columns). Returns (best_regex, best_fitness)."""
    rng = rng or random
    K = len(columns)
    if K == 0:
        return "", -1e9
    pop = [[rng.randrange(GENE_MAX) for _ in range(K)] for _ in range(POP)]
    best = None
    for _ in range(GENERATIONS):
        scored = []
        for g in pop:
            f, rx = fitness(g, columns, members, negatives)
            scored.append((f, g, rx))
        scored.sort(key=lambda x: x[0], reverse=True)
        if best is None or scored[0][0] > best[0]:
            best = (scored[0][0], scored[0][2])
        if scored[0][0] >= -(GAMMA_LEN + GAMMA_C):  # ~perfect (FN=FP=0): early stop
            break
        sel = [(f, g) for f, g, _ in scored]
        nxt = [scored[i][1] for i in range(min(ELITISM, len(scored)))]  # elitism
        while len(nxt) < POP:
            p1, p2 = _tournament(sel), _tournament(sel)
            if rng.random() < P_CROSS and K > 1:                       # crossover
                cut = rng.randrange(1, K)
                child = p1[:cut] + p2[cut:]
            else:
                child = list(p1)
            child = [(rng.randrange(GENE_MAX) if rng.random() < P_MUT else g)  # mutation
                     for g in child]
            nxt.append(child)
        pop = nxt
    return best[1], best[0]


# --- driver ------------------------------------------------------------------
def run_ga(blocked_payloads, negatives=None, guided=False):
    """Extract one generalized regex per blocked-payload cluster.

    blocked_payloads : payloads the WAF blocks (faithful) OR sqlmap-confirmed evasive
                       payloads to direct generalisation (guided=True; declared variant).
    Returns list of dicts: {cluster, size, regex, fitness}.
    """
    negatives = negatives if negatives is not None else DEFAULT_NEGATIVES
    eroded = endpoint_erosion(list(blocked_payloads))
    groups = cluster(eroded)
    out = []
    for lab, members in sorted(groups.items()):
        cols = align_columns(members)
        rx, fit = evolve(cols, members, negatives)
        out.append({"cluster": lab, "size": len(members), "regex": rx,
                    "fitness": round(fit, 4), "guided": guided})
    return out


if __name__ == "__main__":
    import json
    import sys
    payloads = [l.rstrip("\n") for l in sys.stdin] if not sys.stdin.isatty() else []
    if not payloads:
        payloads = [  # demo: a family of UNION-based blocked payloads
            "1' UNION SELECT user,password FROM users-- -",
            "1' UNION SELECT a,b FROM admins-- -",
            "9' UNION SELECT x,y FROM people-- -",
            "5' UNION SELECT c,d FROM members-- -",
            "2' UNION SELECT e,f FROM clients-- -",
            "7' UNION SELECT g,h FROM staff-- -",
        ]
    print(json.dumps(run_ga(payloads), indent=2))
