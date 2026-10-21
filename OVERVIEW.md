# CR-sqlmap — System & Method Overview

*A single-pass review doc: the idea, the architecture, the reinforcement-learning /
deep-learning formulation, the math, and what is actually built vs. pending.*
*Authorized, isolated security-research lab only.*

---

## 1. The idea in one paragraph

CR-sqlmap is a **reinforcement-learning (RL) agent that drives the real `sqlmap`
tool** to find SQL-injection vulnerabilities **and learn to bypass Web
Application Firewalls (WAFs)**. Instead of a human manually trying evasion
tricks (tamper scripts, encodings, timing), an RL agent learns *which
combination of evasion options works against a given WAF*, by trial and error,
rewarded when a payload slips past the WAF and reaches the database. It fuses two
prior works: **Checklist-RL** (Paper 1 — detection efficiency) and **Ouroboros**
(Paper 2 — a dense "symbolic" reward that scores *how close* a payload got, not
just pass/fail), and adds original pieces: an **annealed hybrid reward**, a
**symbolic regex model (FA2RNN)** for the reward, and **circuit breakers** that
detect reward-hacking.

---

## 2. Architecture — the full loop

```
   Kaggle notebook (free GPU)            Kali VM (home LAN)              Ubuntu VM (home LAN)
 ┌───────────────────────────┐      ┌──────────────────────┐      ┌───────────────────────┐
 │  DQN agent  +  BERT enc.   │      │  relay.py (Flask)    │      │  nginx + ModSecurity  │
 │  "the brain" (chooses the  │─────▶│  executor.py         │─────▶│  + OWASP-CRS 942 :8080│
 │  action, learns)           │ HTTPS│  drives sqlmap       │  LAN │  ngx_lua_waf     :8081│
 │                            │ ngrok│  "the hands"         │      │     │                 │
 └───────────────────────────┘      └──────────────────────┘      │     ▼                 │
                                                                   │  DVWA (vuln app)      │
                                                                   └───────────────────────┘
```

- **Ubuntu** = the **target**: DVWA (a deliberately vulnerable app) sitting behind
  two different WAFs — ModSecurity/OWASP-CRS (only the SQLi rule group `942`
  loaded) on `:8080`, and `ngx_lua_waf` on `:8081`. Two WAFs = two different
  detection styles for the agent to learn against.
- **Kali** = the **executor**: `executor.py` wraps `sqlmap`; `relay.py` exposes it
  as an HTTP endpoint (`POST /run`, API-key auth, lab-only target allowlist).
  Kali reaches Ubuntu directly over the LAN.
- **Kaggle** = the **brain**: runs the RL training (needs a GPU). Because Kali is
  behind home NAT (no public IP), an **ngrok tunnel** gives the cloud a public
  URL that routes back into Kali. ngrok is *only* for the Kaggle→Kali hop.

One **training step** = agent picks an action → HTTP call to Kali → sqlmap runs
that attack against a WAF → telemetry (did it bypass? how blocked? timing?) comes
back → reward computed → agent updates. Repeat.

---

## 3. The RL formulation (a Markov Decision Process)

The problem is cast as an MDP $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$.

### 3.1 State $s_t \in \mathbb{R}^{792}$

Concatenation of a **24-dim engineered vector** and a **768-dim BERT embedding**:

- **15 engineered features**: `found`, `block_ratio`, normalized request count,
  normalized duration, `waf_warned`, `not_injectable`, `dbms_known`,
  #techniques, `timed_out`, the last action's (technique, level, risk, operator,
  random-agent), and which WAF.
- **9 SoftChecklist features**: an **exponential moving average** (EMA,
  $\alpha=0.3$) over 9 "blocked-keyword" buckets — a *soft* version of Paper-1's
  binary blocked-keyword checklist:
  $$ \text{EMA}_t = (1-\alpha)\,\text{EMA}_{t-1} + \alpha\, x_t $$
- **768 BERT features**: `bert-base-uncased` CLS embedding of a text summary of
  the last episode (dbms, techniques, payloads, block ratio…). This is the
  **deep-learning encoder** — it turns messy episode telemetry into a dense
  vector the policy network can use.

### 3.2 Action $a_t \in \{0,\dots,755\}$ (756 discrete)

One integer decodes to a full sqlmap configuration:
$$ 756 = \underbrace{9}_{\text{technique}} \times \underbrace{3}_{\text{level}}
\times \underbrace{2}_{\text{risk}} \times \underbrace{7}_{\text{tamper/operator}}
\times \underbrace{2}_{\text{random-agent}} $$
i.e. *which* SQLi techniques (B/E/U/S/T/Q and combos), how hard sqlmap tries
(level, risk), which **tamper/evasion scripts** to apply (none … `space2comment`,
`between`, `randomcase`, `charencode`, combos), and whether to randomize the
User-Agent. The tamper choice is the main evasion lever.

### 3.3 Reward — the heart of the method

Each episode produces a **true outcome** and a **mutated payload** $\chi_{MUT}$
(the base payload $\chi_{ORIG}$ after the chosen tamper chain). Three signals:

**(a) Outcome / system reward** (Ouroboros Eq. 5) — did it actually bypass?
$$ R_{SYS} = \begin{cases} 10 & \text{evasion succeeds (injectable, passed WAF)}\\ 0 & \text{otherwise} \end{cases} $$

**(b) Rule-penetration reward** (Eq. 6) — a **timing side-channel**. Define
*average processing-cost-per-token* $\text{APCT} = T_{exec} / L_{payload}$
(execution time divided by payload length). A mutated payload that makes the WAF
*work harder* (higher APCT) penetrated deeper into its rule set
(validated in the paper: Spearman $\rho = 0.957$ vs. true rule depth):
$$ R_{RUL} = \text{APCT}_{MUT} - \text{APCT}_{ORIG} $$

**(c) Regex-model reward** (Eq. 7) — how much the mutation *fooled a learned
detector* $\psi$ (the FA2RNN, §5), i.e. lowered its "malicious" probability:
$$ R_{REG} = \psi(\chi_{ORIG}) - \psi(\chi_{MUT}) $$

**Symbolic reward** (Eq. 8) — unweighted sum, clipped to $[0,10]$:
$$ R_{SYN} = \min\!\big(10,\ \max(0,\ R_{SYS} + R_{RUL} + R_{REG})\big) $$

**Annealed hybrid** (our addition) — blend the sparse true outcome
$R_{\text{outcome}}=R_{SYS}$ with the dense symbolic reward, annealing the
surrogate weight **down** over training so the agent relies less on the proxy and
more on real success as it learns (this is the anti-reward-hacking schedule):
$$ R_{\text{total}} = (1-\beta_e)\,R_{\text{outcome}} + \beta_e\, R_{SYN},
\qquad \beta_e:\ 0.90 \rightarrow 0.15 $$

> Early training: lean on the dense $R_{SYN}$ to bootstrap (sparse success is rare
> against a strong WAF). Late training: lean on the true outcome $R_{SYS}$ so the
> policy can't "win" by gaming the proxy.

### 3.4 Episode

Start → the agent takes actions; the episode **ends on the first successful
evasion** (`found`) or after `EP_LEN` steps. `ep_len → 1` over training means the
agent learned to bypass on the first try — that *is* the learning signal (reward
can look flat if it's already near the ceiling on an easy WAF).

---

## 4. The learning algorithm — Deep Q-Network (DQN)

The agent learns an **action-value function** $Q(s,a)$ = expected future reward
of taking action $a$ in state $s$. Tabular Q-learning can't handle a
792-dim continuous state, so $Q$ is a **neural network** $Q_\theta$ (an MLP over
the 792-dim state → 756 action-values). Core update (Bellman / temporal
difference):
$$ Q_\theta(s_t,a_t) \leftarrow Q_\theta(s_t,a_t) + \eta\Big[\,
\underbrace{r_t + \gamma \max_{a'} Q_{\theta^-}(s_{t+1},a')}_{\text{TD target}}
- Q_\theta(s_t,a_t)\,\Big] $$
with discount $\gamma = 0.8$, a **target network** $\theta^-$ (a lagged copy, for
stability), an **experience replay buffer** (reuse past transitions), and
**ε-greedy** exploration (ε decays 0.99 → ~0.05, so it explores early and exploits
later). Implemented with **stable-baselines3**; swapping DQN→PPO is one line
(Ouroboros used PPO+DQN) because the environment is a standard Gym env.

---

## 5. FA2RNN — the symbolic regex model behind $R_{REG}$

$\psi$ is a reimplementation of **FA-RNN** (Jiang et al., EMNLP 2020): it turns
WAF **regex rules into a trainable recurrent neural network**, so it is both
interpretable (starts as the exact rules) and learnable. Pipeline:

$$ \text{regex} \xrightarrow{\text{Thompson}} \text{NFA}
\xrightarrow{\text{subset}} \text{DFA} \xrightarrow{\text{minimize}}
\text{weighted FA } (\boldsymbol{\alpha}, \{M_s\}, \boldsymbol{\eta})
\xrightarrow{\text{CP/PARAFAC}} \text{linear RNN} + \text{MLP} \to \psi $$

- A regex compiles to a **weighted finite automaton**: start vector
  $\boldsymbol\alpha$, per-symbol transition matrices $M_s$, accept vector
  $\boldsymbol\eta$. A string is "accepted" iff
  $\boldsymbol\alpha \big(\prod_t M_{s_t}\big)\boldsymbol\eta > 0$.
- The 3-way transition tensor $T\in\mathbb{R}^{|\Sigma|\times n\times n}$ is
  **CP/PARAFAC-decomposed** (rank $R$) into factors $A, D_1, D_2$. This
  "neuralizes" the automaton into a **linear-activation RNN** (FA-RNN Eq. 4):
  $$ h_t = \big((h_{t-1}\, D_1)\odot v_t\big)\, D_2^{\top}, \qquad v_t = A[\,x_t\,] $$
  where $\odot$ is element-wise product and $v_t$ is the embedding of input char
  $x_t$. An **MLP head** maps the final state to the malicious probability
  $\psi = \sigma(\text{MLP}(h_T)) \in (0,1)$.
- Trained positives-only, 70/30 split, 10 epochs, lr 0.01 (paper targets ≈ F1
  0.86). Because multiple `.*`-anchored rules unioned into one DFA blow up
  exponentially, we keep **one automaton per rule** and combine them
  **block-diagonally** (linear in total states).

---

## 6. Guardrails — anti-reward-hacking & cost

Reward hacking = the agent inflates the *proxy* reward ($R_{SYN}$) without real
success (classic Goodhart's Law; Gao et al. 2022 show this "hump-shape" in RLHF).
Two independent **circuit breakers** watch for it:

- **γ (overoptimization)** = normalized **area between the proxy-reward curve and
  the gold-TPR curve** (true success measured on a held-out set with ε=0, every
  N episodes — the "gold-eval loop"). $\gamma\approx 0$ ⇒ proxy tracks reality;
  large γ ⇒ the agent is gaming the proxy. If $\gamma > 0.15$ → **lower β**
  (shift weight back to the true outcome).
- **ρₑ (cost-efficiency)** = marginal true-detection gained per compute-hour:
  $$ \rho_e = \frac{\text{TPR}_e - \text{TPR}_{\text{baseline}}}{\text{compute-hours}_e} $$
  If it trends to 0 / negative, extra compute isn't buying accuracy → stop early.
- Plus a hard **compute-runaway** cap (max requests / wall-clock) so an
  unattended run can't overrun.

---

## 7. Evaluation — metric Groups A–D (5 seeds, mean ± 95% CI)

| Group | Measures | Metrics |
|---|---|---|
| **A** detection quality | does it find vulns, efficiently? | TPR, FPR, Precision, F1, Requests-per-Vuln, Requests-per-Safe, Runtime |
| **B** evasion + transfer | does it bypass, and generalize across WAFs? | TSR, FNR, cross-WAF **transfer matrix** $\text{TransferRate}(A\to B)$, Spearman ρ(APCT↔rule-depth) |
| **C** payload quality / phase | diverse payloads, or collapsing to one trick? | Duplicate-Rate (MD5), Type-Token-Ratio, Shannon entropy — **per phase** (early/mid/late) |
| **D** CR-sqlmap-specific | do the monitors fire correctly? | γ (proxy–gold divergence), FA2RNN covariate shift, ρₑ |

Transfer rate: $\text{TransferRate}(A\!\to\!B) = \dfrac{|\{x: x \text{ bypasses } A \text{ and } B\}|}{|\{x: x \text{ bypasses } A\}|}$.

---

## 8. Code map

| file | role |
|---|---|
| `executor.py` | runs one sqlmap episode → JSON telemetry; builds the χ_ORIG/χ_MUT APCT probe |
| `relay.py` | Flask relay on Kali (API-key auth, lab-only targets, bounded concurrency) |
| `kaggle_waf_rl.py` | Gym env + DQN + 792-d state + annealed reward + gold-eval + N parallel envs |
| `circuit_breakers.py` | γ (anti-reward-hacking) + compute-runaway breakers |
| `fa2rnn.py` | the FA-RNN model for $R_{REG}$ (regex → weighted-FA → linear RNN → ψ) |
| `eval.py` | metric Groups A–D, multi-seed CI |
| `start_lab.sh` | Kali keepalive: relay + ngrok (idempotent; supports a stable NGROK_DOMAIN) |
| `kaggle_runner.py` | optional two-account Kaggle rotation (run with your own creds) |

---

## 9. How to run (the full circle)

1. **Kali**: `./start_lab.sh` → prints the ngrok URL; `grep RELAY_API_KEY relay.env` → the key.
2. **Kaggle**: Secrets `RELAY_URL` + `RELAY_API_KEY`; GPU + Internet ON; then:
   ```python
   !git clone https://github.com/MinhBe/cr-sqlmap.git && cd cr-sqlmap
   !pip -q install -r requirements.txt
   !WAF=modsec N_ENVS=2 python -u kaggle_waf_rl.py
   ```
3. Checkpoints + `eval_log.json` → `/kaggle/working/ckpt`; evaluate with `python eval.py run`.

---

## 10. Status — verified vs. pending

**Verified working**
- Full loop live end-to-end (Kaggle → ngrok → Kali → sqlmap → WAF → DVWA → back).
- DQN **learning confirmed** on `luawaf`: `ep_len` collapsed 4.7 → ~1.1 (learns to bypass in one step).
- `R_SYS` + `R_RUL` **live** in the reward; β anneals 0.90 → 0.15; gold-eval loop writes `eval_log.json`.
- Circuit breakers, parallel envs (N=2 proven 2× on Kali), eval math (Groups A–D) all self-tested.
- Two WAFs confirmed with different profiles (ModSecurity blocks ~92–95%; ngx_lua_waf keyword-based).

**Pending / to tune (your hands)**
- Run `WAF=modsec` long enough to see the real learning curve (reward climbing from ~0).
- `R_REG` turns on once the FA2RNN is **trained on Kaggle** and loaded (`env.fa2rnn`).
- Group B transfer matrix + Group D covariate-shift need the second-WAF bypass sets / a trained FA2RNN.
- Tunables to hand-check vs. your master doc: `R_RUL_SCALE` (=1000), `APCT_PROBES` (=3), γ threshold (0.15), β schedule.
- Hardening: rotate the relay API key; reserve a static ngrok domain (or switch to Cloudflare Tunnel) for a permanent URL.
