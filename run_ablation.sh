#!/usr/bin/env bash
# Ablation: cold-start (random init, Ouroboros baseline) vs warm-start
# (behavior-cloning pretrain) under identical seeds/budget. Run on Kaggle.
#   bash run_ablation.sh          # defaults: modsec, seeds 42 123 456, 5000 steps
# Each run writes its own ckpt/<cold|warm>_<seed>/ (checkpoints + tb + eval_log.json)
# so the learning curves can be compared directly.
set -u
WAF=${WAF:-modsec}
SEEDS=${SEEDS:-"42 123 456"}
STEPS=${TOTAL_TIMESTEPS:-5000}
N=${N_ENVS:-2}
BC=${BC_DATASET:-bc_dataset.npz}

# 1) Collect the behavior-cloning dataset once (successful (state,action) pairs).
#    Default source = luawaf (successes are frequent -> a transferable evasion prior).
if [ ! -f "$BC" ]; then
    echo "===== collecting BC dataset -> $BC ====="
    BC_WAF=${BC_WAF:-luawaf} python -u warmstart.py collect --out "$BC" --n ${BC_N:-500}
fi

# 2) For each seed: cold then warm, identical budget.
for S in $SEEDS; do
    echo "===== seed $S : COLD-START (random init) ====="
    CKPT_DIR="ckpt/cold_$S" SEED="$S" WARM_START=0 \
        WAF="$WAF" N_ENVS="$N" TOTAL_TIMESTEPS="$STEPS" python -u kaggle_waf_rl.py

    echo "===== seed $S : WARM-START (BC pretrain) ====="
    CKPT_DIR="ckpt/warm_$S" SEED="$S" WARM_START=1 BC_DATASET="$BC" \
        WAF="$WAF" N_ENVS="$N" TOTAL_TIMESTEPS="$STEPS" python -u kaggle_waf_rl.py
done

# 3) Compare (Group A success-rate + requests-per-vuln per run):
echo "===== comparison ====="
for S in $SEEDS; do
    for arm in cold warm; do
        echo "--- $arm seed $S ---"
        CKPT_DIR="ckpt/${arm}_$S" WAF="$WAF" python eval.py run 2>/dev/null | \
            python -c "import sys,json; d=json.load(sys.stdin); print(d.get('groupA_summary'))" 2>/dev/null || echo "(eval skipped)"
    done
done
echo "Curves: tensorboard ckpt/*/tb ; gold curve: ckpt/*/eval_log.json ; live: [ep step=] lines."
