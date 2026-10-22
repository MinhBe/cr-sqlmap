#!/usr/bin/env python3
"""
warmstart.py - behavior-cloning warm-start for the CR-sqlmap DQN (ablation:
cold-start random init vs warm-start BC pretrain). Isolated lab only.

Two pieces:
  1) collect(): run the env and log (state, action) pairs for the actions that
     SUCCESSFULLY evaded the WAF (telemetry.found == True) -> a BC dataset.
  2) behavior_clone(): supervised pretrain of the DQN's Q-network to predict the
     successful action from the state (cross-entropy over the 756 discrete
     actions, Q-values as logits), then copy weights into the target net. Call it
     after building the SB3 model and before model.learn() -> warm start.

Design choices (delegated; hand-tune later):
  - BC loss = CrossEntropy(Q_theta(s), a*)   (standard discrete BC)
  - collect on BC_WAF (default luawaf: successes are frequent, giving a
    transferable evasion prior for the harder modsec target)
  - epochs=20, lr=1e-3, Adam, batch=256
torch/SB3 run on Kaggle; this file is imported lazily so the rest stays testable.
"""
import argparse
import os


def collect(waf=None, n_success=500, out_path="bc_dataset.npz", seed=0, max_steps=20000):
    """Collect `n_success` successful (state, action) pairs with a random policy."""
    import numpy as np
    import kaggle_waf_rl as K
    waf = waf or os.environ.get("BC_WAF", "luawaf")
    env = K.make_env(waf=waf)
    obs, _ = env.reset(seed=seed)
    X, Y = [], []
    steps = 0
    while len(X) < n_success and steps < max_steps:
        a = env.action_space.sample()
        prev = obs
        obs, _, term, trunc, info = env.step(int(a))
        steps += 1
        if info.get("telemetry", {}).get("found"):
            X.append(np.asarray(prev, dtype=np.float32)); Y.append(int(a))
        if term or trunc:
            obs, _ = env.reset()
        if steps % 25 == 0:
            print(f"[bc-collect] waf={waf} steps={steps} collected={len(X)}/{n_success}", flush=True)
    X = np.asarray(X, dtype=np.float32); Y = np.asarray(Y, dtype=np.int64)
    np.savez(out_path, X=X, Y=Y)
    print(f"[bc-collect] saved {len(X)} pairs -> {out_path} (unique actions: {len(set(Y.tolist()))})", flush=True)
    return out_path


def behavior_clone(model, dataset_path, epochs=20, lr=1e-3, batch=256):
    """Pretrain model.policy.q_net on the BC dataset (CE), then sync target net."""
    import numpy as np
    import torch
    data = np.load(dataset_path)
    X = torch.as_tensor(data["X"], dtype=torch.float32)
    Y = torch.as_tensor(data["Y"], dtype=torch.long)
    if len(X) == 0:
        print("[bc] empty dataset - skipping warm start", flush=True)
        return model
    dev = model.device
    qnet = model.policy.q_net
    opt = torch.optim.Adam(qnet.parameters(), lr=lr)
    ce = torch.nn.CrossEntropyLoss()
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot, correct, last = 0.0, 0, 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = X[idx].to(dev); yb = Y[idx].to(dev)
            logits = qnet(xb)                      # Q-values used as action logits
            loss = ce(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            last = loss.item(); tot += last
            correct += (logits.argmax(1) == yb).sum().item()
        print(f"[bc] epoch {ep+1}/{epochs} loss={tot/max(1,(n//batch+1)):.3f} "
              f"train_acc={correct/n:.3f}", flush=True)
    # warm-start the target network too
    model.policy.q_net_target.load_state_dict(model.policy.q_net.state_dict())
    print("[bc] warm-start done: q_net + target initialized from BC", flush=True)
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    c = sub.add_parser("collect")
    c.add_argument("--waf", default=os.environ.get("BC_WAF", "luawaf"))
    c.add_argument("--n", type=int, default=int(os.environ.get("BC_N", "500")))
    c.add_argument("--out", default=os.environ.get("BC_DATASET", "bc_dataset.npz"))
    c.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "collect":
        collect(waf=args.waf, n_success=args.n, out_path=args.out, seed=args.seed)
    else:
        ap.print_help()
