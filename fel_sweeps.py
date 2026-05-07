#!/usr/bin/env python3
"""
fel_sweeps.py — minimal-edit sweep runner for your FEL repo.

Usage examples (from project root):
  python fel_sweeps.py --base-config configs/example.yaml --out-root runs/fel_sweeps --study 1
  python fel_sweeps.py --base-config configs/example.yaml --out-root runs/fel_sweeps --study 2 --clients RSNA,MENDELEY
  python fel_sweeps.py --base-config configs/example.yaml --out-root runs/fel_sweeps --study 4 --scenario skew_3to1 --label-noise 0.05

  # NEW — Experiment 1: rigorous eval framework + centralized upper bound
  python fel_sweeps.py --base-config configs/example.yaml --out-root runs/fel_sweeps --study 0

The script:
- Loads your base YAML (keeps all dataset paths as-is).
- Writes derived YAMLs per run (seed_k, AL method, K, META, RF, etc.).
- Computes rounds from desired label budgets (seed_k + rounds*B).
- Calls: PYTHONPATH=src python tools/run_fel.py --config <derived.yaml>
- Stores each run under out-root/<study_id>/<short_name>/ with its YAML and JSONL logs.

Study 0 (NEW):
- Runs the federated pipeline with a 30% stratified holdout test set.
- Runs a centralized (pooled) upper-bound baseline with the same holdout.
- Sweeps the same AL methods as the federated run for direct comparison.
"""
import argparse, os, sys, subprocess, itertools, time, shutil, math, json
from pathlib import Path
from datetime import datetime

try:
    import yaml  # PyYAML
except Exception as e:
    print("PyYAML is required. pip install pyyaml", file=sys.stderr); raise

def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def dump_yaml(obj, p):
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def derive_rounds(seed_k:int, add_labels:int, batch_B:int) -> int:
    """Compute number of rounds to add ~add_labels (ceil(add_labels / B))."""
    return max(1, math.ceil(add_labels / max(1, batch_B)))

def set_cfg(cfg, updates:dict):
    """Shallow set helper for known keys."""
    for k,v in updates.items():
        node = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v
    return cfg

def ensure_logging_dir(cfg, out_dir):
    cfg = set_cfg(cfg, {"logging.out_dir": out_dir})
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return cfg

def run_one(base_cfg_path, cfg_updates, out_dir, tag, runner="tools/run_fel.py"):
    cfg = load_yaml(base_cfg_path)
    cfg = set_cfg(cfg, cfg_updates)
    cfg = ensure_logging_dir(cfg, out_dir)
    # Write derived yaml
    ypath = Path(out_dir) / f"derived_{tag}.yaml"
    dump_yaml(cfg, ypath)
    # Launch
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    cmd = ["python", runner, "--config", str(ypath)]
    print(">>> RUN:", " ".join(cmd))
    print(">>> OUT:", out_dir)
    subprocess.run(cmd, env=env, check=True)


# =========================================================================
#  Study 0 — Experiment 1: Rigorous Evaluation Framework
# =========================================================================

def study_eval(
    base,
    out_root,
    seed_k=200,
    add_labels=12000,
    batch_B=200,
    holdout_frac=0.30,
    methods=("least_confident", "margin", "entropy", "bald", "qbc"),
    K=0,
):
    """
    Experiment 1: Rigorous evaluation with fixed holdout + centralized oracle.

    For each AL method:
      1. Run the **federated** pipeline with a 30% stratified holdout test set.
         Metrics are logged both on the shrinking U (backward compat) and the
         fixed test set.
      2. Run the **centralized** (pooled) upper-bound baseline with the same
         holdout and AL method, producing the oracle ceiling line.

    All runs share the same random seed → identical holdout split.
    """
    rounds = derive_rounds(seed_k, add_labels, batch_B)

    for m in methods:
        # ---------- A) Federated run with holdout ----------
        tag_fed = f"eval_fed_{m}_K{K}_seed{seed_k}_B{batch_B}_R{rounds}_holdout{int(holdout_frac*100)}"
        out_fed = os.path.join(out_root, "0_eval_framework", tag_fed)
        if not os.path.exists(out_fed):
            updates = {
                "seed_k": seed_k,
                "rounds": rounds,
                "al.method": m,
                "al.batch_B": batch_B,
                "augment.train_n_per_sample": K,
                "meta.aug_mode": "off",
                "meta.refresh_every": 1,
                # NEW: enable holdout
                "holdout.test_frac": holdout_frac,
            }
            run_one(base, updates, out_fed, tag_fed, runner="tools/run_fel.py")

        # ---------- B) Centralized upper-bound ----------
        tag_cen = f"eval_central_{m}_K{K}_seed{seed_k}_B{batch_B}_R{rounds}_holdout{int(holdout_frac*100)}"
        out_cen = os.path.join(out_root, "0_eval_framework", tag_cen)
        if not os.path.exists(out_cen):
            updates = {
                "seed_k": seed_k,
                "rounds": rounds,
                "al.method": m,
                "al.batch_B": batch_B,
                "augment.train_n_per_sample": K,
                "meta.aug_mode": "off",
                "meta.refresh_every": 1,
                "holdout.test_frac": holdout_frac,
            }
            run_one(base, updates, out_cen, tag_cen, runner="tools/run_centralized.py")


# =========================================================================
#  Existing studies (1–4) unchanged
# =========================================================================

def study1(base, out_root, n_estimators=(100, 500, 1000, 1500), max_features=("sqrt","log2"), batch_B=200, seed_k=200, add_labels=12000, K=0):
    rounds = derive_rounds(seed_k, add_labels, batch_B)
    for ne in n_estimators:
        for mf in max_features:
            tag = f"study1_bald_ne{ne}_mf{mf}_K{K}_seed{seed_k}_B{batch_B}_R{rounds}"
            out_dir = os.path.join(out_root, "1_committee", tag)
            if os.path.exists(out_dir):
                continue
            updates = {
                "seed_k": seed_k,
                "rounds": rounds,
                "al.method": "bald",
                "al.batch_B": batch_B,
                "augment.train_n_per_sample": K,
                "rf.n_estimators": ne,
                "rf.max_features": mf,
                "meta.aug_mode": "off",
                "meta.refresh_every": 1,
            }
            run_one(base, updates, out_dir, tag)

def study2(base, out_root, scenario="baseline", per_class_min_vals=(5,0), batch_B=100, seed_k=200, add_labels=5000, K=5, tta_row=False):
    rounds = derive_rounds(seed_k, add_labels, batch_B)
    for pcm in per_class_min_vals:
        tag = f"study2_{scenario}_pcm{pcm}_K{K}_seed{seed_k}_B{batch_B}_R{rounds}{'_tta3' if tta_row else ''}"
        out_dir = os.path.join(out_root, "2_noniid", tag)
        if os.path.exists(out_dir):
            continue
        updates = {
            "seed_k": seed_k,
            "rounds": rounds,
            "al.method": "bald",
            "al.batch_B": batch_B,
            "augment.train_n_per_sample": K,
            "meta.aug_mode": ("tta" if tta_row else "off"),
            "meta.tta_n": (3 if tta_row else 0),
            "meta.refresh_every": 1,
        }
        run_one(base, updates, out_dir, tag)

def study3(base, out_root, seed_k=200, add_labels=12000, batch_B_list=(50,100,200), K=0, lambdas=(1e-3,1e-2,1e-1,1,1e1,1e2,1e3)):
    rounds_map = {B: derive_rounds(seed_k, add_labels, B) for B in batch_B_list}
    for method in ("k_coreset","kmeans"):
        tag = f"study3_seedmethod_{method}_K{K}_seed{seed_k}_B100_R{rounds_map.get(100,derive_rounds(seed_k,add_labels,100))}"
        out_dir = os.path.join(out_root, "3_extras", tag)
        if os.path.exists(out_dir):
            continue
        updates = {
            "seed_k": seed_k,
            "rounds": rounds_map.get(100, derive_rounds(seed_k, add_labels, 100)),
            "seeding.method": method,
            "al.method": "bald",
            "al.batch_B": 100,
            "augment.train_n_per_sample": K,
            "meta.aug_mode": "off",
            "meta.refresh_every": 1,
        }
        run_one(base, updates, out_dir, tag)
    for B in batch_B_list:
        tag = f"study3_batch_B{B}_K{K}_seed{seed_k}_R{rounds_map[B]}"
        out_dir = os.path.join(out_root, "3_extras", tag)
        if os.path.exists(out_dir):
            continue
        updates = {
            "seed_k": seed_k,
            "rounds": rounds_map[B],
            "al.method": "bald",
            "al.batch_B": B,
            "augment.train_n_per_sample": K,
            "meta.aug_mode": "off",
            "meta.refresh_every": 1,
        }
        run_one(base, updates, out_dir, tag)
    for lam in lambdas:
        tag = f"study3_meta_lambda_{lam}_K{K}_seed{seed_k}_B100_R{rounds_map.get(100,derive_rounds(seed_k,add_labels,100))}"
        out_dir = os.path.join(out_root, "3_extras", tag)
        if os.path.exists(out_dir):
            continue
        updates = {
            "seed_k": seed_k,
            "rounds": rounds_map.get(100, derive_rounds(seed_k, add_labels, 100)),
            "al.method": "bald",
            "al.batch_B": 100,
            "augment.train_n_per_sample": K,
            "meta.lambda": float(lam),
            "meta.aug_mode": "off",
            "meta.refresh_every": 1,
        }
        run_one(base, updates, out_dir, tag)

def study4(base, out_root, seed_k=200, add_labels=12000, batch_B=200,
           methods=("bald","qbc","entropy","margin","least_confident"),
           n_aug=(0,8,16,24,32), meta_modes=("off","tta"),
           tta_grid=(4,8,16,24,32), refresh_every=1):
    rounds = derive_rounds(seed_k, add_labels, batch_B)
    for m in methods:
        for K in n_aug:
            for mode in meta_modes:
                tta_values = [0] if (mode == "off" or K == 0) else list(tta_grid)
                if K == 0 and mode == "tta":
                    continue
                if len(tta_values) > 1:
                    tta_values = [tta for tta in tta_values if tta <= K]
                for tta_n in tta_values:
                    tag = (
                        f"study4_{m}_K{K}_{mode}"
                        f"{'' if mode=='off' else f'_tta{tta_n}'}"
                        f"_seed{seed_k}_B{batch_B}_R{rounds}"
                    )
                    out_dir = os.path.join(out_root, "4_sampler_aug", tag)
                    if os.path.exists(out_dir):
                        continue
                    updates = {
                        "seed_k": seed_k,
                        "rounds": rounds,
                        "al.method": m,
                        "al.batch_B": batch_B,
                        "augment.train_n_per_sample": K,
                        "meta.aug_mode": mode,
                        "meta.tta_n": (0 if mode == "off" else int(tta_n)),
                        "meta.refresh_every": int(refresh_every),
                    }
                    run_one(base, updates, out_dir, tag)


# =========================================================================
#  Study 5 — Experiment 2: Scalability and Non-IID Robustness
# =========================================================================

def study_scalability(
    base,
    out_root,
    add_labels=18000,
    batch_B=200,
    holdout_frac=0.15,
    al_method="margin",          # best from Exp 1
    K=0,
    n_clients_list=(2, 5, 10, 20, 30),
    alpha_list=(0.1, 0.5, 1.0),
):
    """
    Experiment 2: Scalability and Non-IID Robustness.

    Factorial sweep over C (number of simulated clients) × α (Dirichlet
    concentration) plus a uniform (IID) control per C.
    Uses the same fixed holdout test set as Exp 1.

    seed_k scales with C to avoid over-seeding small per-client pools:
        C=2 → 200,  C=5 → 100,  C=10 → 50,  C=20 → 20,  C=30 → 10
    """
    # Seed budget per C — keeps seed ≤ 2-5% of expected per-client pool
    seed_k_map = {2: 200, 5: 100, 10: 50, 20: 20, 30: 10}

    for C in n_clients_list:
        seed_k = seed_k_map.get(C, max(10, 200 // C))
        rounds = derive_rounds(seed_k, add_labels, batch_B)

        # Build the list of (alpha_value, partition_method) to run
        configs = []
        if C == 2:
            # Natural 2-client baseline; no partitioning needed
            configs.append((None, None))
        else:
            # Dirichlet sweeps
            for alpha in alpha_list:
                configs.append((alpha, "dirichlet"))
            # Uniform (IID) control
            configs.append((None, "uniform"))

        for alpha, method in configs:
            if method is None and C == 2:
                tag = (f"exp2_C{C}_{al_method}_K{K}"
                       f"_seed{seed_k}_B{batch_B}_R{rounds}_holdout{int(holdout_frac*100)}")
            elif method == "uniform":
                tag = (f"exp2_C{C}_uniform_{al_method}_K{K}"
                       f"_seed{seed_k}_B{batch_B}_R{rounds}_holdout{int(holdout_frac*100)}")
            else:
                tag = (f"exp2_C{C}_alpha{alpha}_{al_method}_K{K}"
                       f"_seed{seed_k}_B{batch_B}_R{rounds}_holdout{int(holdout_frac*100)}")

            out_dir = os.path.join(out_root, "2_scalability", tag)
            if os.path.exists(out_dir):
                continue

            updates = {
                "seed_k": seed_k,
                "rounds": rounds,
                "al.method": al_method,
                "al.batch_B": batch_B,
                "augment.train_n_per_sample": K,
                "meta.aug_mode": "off",
                "meta.refresh_every": 1,
                "holdout.test_frac": holdout_frac,
            }

            # Add partition config for C > 2
            if method is not None:
                updates["partition.method"] = method
                updates["partition.n_clients"] = C
                if alpha is not None:
                    updates["partition.alpha"] = alpha

            run_one(base, updates, out_dir, tag, runner="tools/run_fel.py")


# =========================================================================
#  Study 6 — Experiment 3: Meta-Feature Ablation + Calibration Quality
# =========================================================================

def study_meta_ablation(
    base,
    out_root,
    seed_k=200,
    add_labels=12000,
    batch_B=200,
    holdout_frac=0.30,
    al_method="margin",
    K=0,
    feature_configs=None,
    lambdas=(1e-3, 1e-2, 1e-1, 1, 10, 100, 1000),
):
    """
    Experiment 3: Meta-Feature Ablation + Calibration Quality.

    Two sub-sweeps on the C=2 natural federation:

    A) Feature-group ablation (progressive addition):
       1. mean_p only  (RF committee mean probs — baseline)
       2. mean_p + var_p  (add per-class variance)
       3. mean_p + var_p + disagreement  (add JS + variation ratio)
       4. all  (add entropy + top-2 margin)

    B) Ridge lambda sweep with full features to check regularization
       sensitivity.

    ECE and Brier are logged every round alongside macro-F1.
    """
    if feature_configs is None:
        feature_configs = [
            ["mean_p"],
            ["mean_p", "var_p"],
            ["mean_p", "var_p", "disagreement"],
            ["mean_p", "var_p", "disagreement", "uncertainty"],
        ]

    rounds = derive_rounds(seed_k, add_labels, batch_B)

    # --- A) Feature-group ablation (lambda fixed at default 1e-2) ---
    for fg in feature_configs:
        fg_label = "+".join(fg)
        tag = (f"exp3_feat_{fg_label}_{al_method}_K{K}"
               f"_seed{seed_k}_B{batch_B}_R{rounds}_holdout{int(holdout_frac*100)}")
        out_dir = os.path.join(out_root, "3_meta_ablation", tag)
        if os.path.exists(out_dir):
            continue

        updates = {
            "seed_k": seed_k,
            "rounds": rounds,
            "al.method": al_method,
            "al.batch_B": batch_B,
            "augment.train_n_per_sample": K,
            "meta.aug_mode": "off",
            "meta.refresh_every": 1,
            "meta.feature_groups": ",".join(fg),
            "holdout.test_frac": holdout_frac,
        }
        run_one(base, updates, out_dir, tag, runner="tools/run_fel.py")

    # --- B) Ridge lambda sweep (all features) ---
    for lam in lambdas:
        lam_str = f"{lam:.0e}" if lam >= 1 else f"{lam}"
        tag = (f"exp3_lambda_{lam_str}_{al_method}_K{K}"
               f"_seed{seed_k}_B{batch_B}_R{rounds}_holdout{int(holdout_frac*100)}")
        out_dir = os.path.join(out_root, "3_meta_ablation", tag)
        if os.path.exists(out_dir):
            continue

        updates = {
            "seed_k": seed_k,
            "rounds": rounds,
            "al.method": al_method,
            "al.batch_B": batch_B,
            "augment.train_n_per_sample": K,
            "meta.aug_mode": "off",
            "meta.refresh_every": 1,
            "meta.lambda": float(lam),
            "meta.feature_groups": "mean_p,var_p,disagreement,uncertainty",
            "holdout.test_frac": holdout_frac,
        }
        run_one(base, updates, out_dir, tag, runner="tools/run_fel.py")


# =========================================================================
#  Study 7 — Experiment 3b: Heterogeneous RF Configurations
# =========================================================================

def study_meta_hetero_rf(
    base,
    out_root,
    seed_k=200,
    add_labels=12000,
    batch_B=200,
    holdout_frac=0.30,
    al_method="margin",
    K=0,
):
    """
    Experiment 3b: Does META help when clients have heterogeneous RF configs?

    Clients are sorted alphabetically (MENDELEY=idx0, RSNA=idx1).
    Since RSNA is more diverse, we test both directions:
      - weak_rsna:     MENDELEY=strong, RSNA=weak  (harder: diverse data + weak model)
      - weak_mendeley: MENDELEY=weak, RSNA=strong   (easier: simple data + weak model)

    Levels of heterogeneity: mild / moderate / extreme.
    Plus one homogeneous baseline.
    """
    import json as _json
    rounds = derive_rounds(seed_k, add_labels, batch_B)

    strong = {"n_estimators": 200, "max_features": "sqrt"}
    weak_configs = {
        "mild":     {"n_estimators": 50, "max_depth": 10, "max_features": "sqrt"},
        "moderate": {"n_estimators": 20, "max_depth": 5,  "max_features": "sqrt"},
        "extreme":  {"n_estimators": 10, "max_depth": 3,  "max_features": "sqrt"},
    }

    # (tag_label, [MENDELEY_cfg, RSNA_cfg])
    levels = [
        ("homogeneous", [
            {"n_estimators": 100, "max_features": "sqrt"},
            {"n_estimators": 100, "max_features": "sqrt"},
        ]),
    ]
    # Both directions for each heterogeneity level
    for severity, weak_cfg in weak_configs.items():
        # RSNA (diverse) gets weak model — harder, META should help more
        levels.append((f"{severity}_weakRSNA", [dict(strong), dict(weak_cfg)]))
        # MENDELEY (simpler) gets weak model
        levels.append((f"{severity}_weakMEND", [dict(weak_cfg), dict(strong)]))

    for level_name, rf_configs in levels:
        tag = (f"exp3b_hetero_{level_name}_{al_method}_K{K}"
               f"_seed{seed_k}_B{batch_B}_R{rounds}_holdout{int(holdout_frac*100)}")
        out_dir = os.path.join(out_root, "3b_hetero_rf", tag)
        if os.path.exists(out_dir):
            continue

        updates = {
            "seed_k": seed_k,
            "rounds": rounds,
            "al.method": al_method,
            "al.batch_B": batch_B,
            "augment.train_n_per_sample": K,
            "meta.aug_mode": "off",
            "meta.refresh_every": 1,
            "meta.feature_groups": "mean_p,var_p,disagreement,uncertainty",
            "holdout.test_frac": holdout_frac,
            "heterogeneous_rf": _json.dumps(rf_configs),
        }
        run_one(base, updates, out_dir, tag, runner="tools/run_fel.py")


# =========================================================================
#  Study 8 — Experiment 3c: Client-Specific Embedding Noise
# =========================================================================

def study_meta_noise(
    base,
    out_root,
    seed_k=200,
    add_labels=12000,
    batch_B=200,
    holdout_frac=0.30,
    al_method="margin",
    K=0,
):
    """
    Experiment 3c: Does META help when clients have different noise levels?

    IID 5-client partition (matching 3d structure). Simulates different
    imaging protocols / scanner quality by adding persistent Gaussian
    noise to client embeddings.

    Clients sorted alphabetically after partition:
      MENDELEY_0, MENDELEY_1, MENDELEY_2, RSNA_0, RSNA_1

    Noise levels (5 values, one per client):
      clean:       all zero                     — baseline
      one_noisy:   one client degraded (σ=0.3)  — single bad scanner
      one_heavy:   one client heavily degraded   — worst-case single site
      gradient:    progressive 0→0.5             — varying scanner quality
      half_noisy:  3 clean + 2 noisy            — mixed federation
      all_mild:    uniform mild noise            — all scanners slightly off
      all_heavy:   uniform heavy noise           — all scanners degraded

    Uses mean_p + uncertainty features (7d) to match Exp 3d.
    """
    import json as _json
    rounds = derive_rounds(seed_k, add_labels, batch_B)

    # [MENDELEY_0, MENDELEY_1, MENDELEY_2, RSNA_0, RSNA_1]
    levels = [
        ("clean",       [0.0, 0.0, 0.0, 0.0, 0.0]),
        ("one_noisy",   [0.0, 0.0, 0.0, 0.0, 0.3]),
        ("one_heavy",   [0.0, 0.0, 0.0, 0.0, 0.5]),
        ("gradient",    [0.0, 0.1, 0.2, 0.3, 0.5]),
        ("half_noisy",  [0.0, 0.0, 0.0, 0.3, 0.5]),
        ("all_mild",    [0.1, 0.1, 0.1, 0.1, 0.1]),
        ("all_heavy",   [0.5, 0.5, 0.5, 0.5, 0.5]),
    ]

    for level_name, noise_stds in levels:
        tag = (f"exp3c_noise_{level_name}_{al_method}_K{K}"
               f"_seed{seed_k}_B{batch_B}_R{rounds}_holdout{int(holdout_frac*100)}")
        out_dir = os.path.join(out_root, "3c_emb_noise", tag)
        if os.path.exists(out_dir):
            continue

        updates = {
            "seed_k": seed_k,
            "rounds": rounds,
            "al.method": al_method,
            "al.batch_B": batch_B,
            "augment.train_n_per_sample": K,
            "meta.aug_mode": "off",
            "meta.refresh_every": 1,
            "meta.feature_groups": "mean_p,uncertainty",
            "holdout.test_frac": holdout_frac,
            # IID 5-client partition (matches 3d)
            "partition.method": "uniform",
            "partition.n_clients": 5,
            "emb_noise_stds": _json.dumps(noise_stds),
        }
        run_one(base, updates, out_dir, tag, runner="tools/run_fel.py")


# =========================================================================
#  Study 9 — Experiment 3d: Mixed-Architecture Federation
# =========================================================================

def study_hetero_models(
    base,
    out_root,
    seed_k=200,
    add_labels=12000,
    batch_B=200,
    holdout_frac=0.30,
    al_method="margin",
    K=0,
):
    """
    Experiment 3d: Does META help with genuinely heterogeneous model types?

    IID 5-client partition. Each client gets a different classifier:
      Client 0: RF         (smooth averaged votes, well-calibrated)
      Client 1: Logistic   (linear, Platt-calibrated by construction)
      Client 2: SVM (RBF)  (sharp Platt-scaled sigmoids)
      Client 3: MLP        (2-layer, softmax, typically overconfident)
      Client 4: GBT        (stagewise, sharper than RF)

    META uses mean_p + uncertainty features (K+2 = 7 dims) — the
    model-agnostic feature set.

    Control: all 5 clients use RF (same architecture, IID partition).
    """
    import json as _json
    rounds = derive_rounds(seed_k, add_labels, batch_B)

    configs = [
        # Mixed architecture federation
        ("mixed",  ["rf", "logistic", "svm", "mlp", "gbt"]),
    ]

    for tag_label, model_types in configs:
        tag = (f"exp3d_{tag_label}_{al_method}_K{K}"
               f"_seed{seed_k}_B{batch_B}_R{rounds}_holdout{int(holdout_frac*100)}")
        out_dir = os.path.join(out_root, "3d_hetero_models", tag)
        if os.path.exists(out_dir):
            continue

        updates = {
            "seed_k": seed_k,
            "rounds": rounds,
            "al.method": al_method,
            "al.batch_B": batch_B,
            "augment.train_n_per_sample": K,
            "meta.aug_mode": "off",
            "meta.refresh_every": 1,
            "meta.feature_groups": "mean_p,uncertainty",
            "holdout.test_frac": holdout_frac,
            # IID 5-client partition
            "partition.method": "uniform",
            "partition.n_clients": 5,
            # Model types
            "client_model_types": _json.dumps(model_types),
        }
        run_one(base, updates, out_dir, tag, runner="tools/run_fel.py")


# =========================================================================
#  Study 10 — Experiment 5: Comparison with Federated Baselines
# =========================================================================

def study_fed_baselines(
    base,
    out_root,
    seed_k=200,
    add_labels=12000,
    batch_B=200,
    holdout_frac=0.30,
    al_method="margin",
    K=0,
):
    """
    Experiment 5: Compare FRML with FedAvg, FedProx, FedNova.

    All methods use the same C=2 natural federation, same holdout,
    same seed, same AL budget (margin sampling).

    Fair comparison with controlled variables:
      FRML+RF:   RF local + ridge meta, communicates Sxx/Sxy (~484 bytes)
      FRML+MLP:  MLP local + ridge meta, communicates Sxx/Sxy (~484 bytes)
      FedAvg:    MLP local + weight averaging (~75 KB)
      FedProx:   MLP local + proximal weight averaging (~75 KB)
      FedNova:   MLP local + step-normalized averaging (~75 KB)

    FRML+MLP vs FedAvg/FedProx/FedNova isolates the federation method
    while holding the local model constant (MLP).
    """
    import json as _json
    rounds = derive_rounds(seed_k, add_labels, batch_B)

    # --- FRML runs (via sim_runner) ---
    frml_runs = [
        # FRML with RF (existing Exp 1 design, re-run for consistency)
        ("frml_rf", {
            "seed_k": seed_k,
            "rounds": rounds,
            "al.method": al_method,
            "al.batch_B": batch_B,
            "augment.train_n_per_sample": K,
            "meta.aug_mode": "off",
            "meta.refresh_every": 1,
            "meta.feature_groups": "mean_p,uncertainty",
            "holdout.test_frac": holdout_frac,
        }),
        # FRML with MLP (same ridge meta, but MLP local model)
        ("frml_mlp", {
            "seed_k": seed_k,
            "rounds": rounds,
            "al.method": al_method,
            "al.batch_B": batch_B,
            "augment.train_n_per_sample": K,
            "meta.aug_mode": "off",
            "meta.refresh_every": 1,
            "meta.feature_groups": "mean_p,uncertainty",
            "holdout.test_frac": holdout_frac,
            "client_model_types": _json.dumps(["mlp", "mlp"]),
        }),
    ]

    for method_name, updates in frml_runs:
        tag = (f"exp5_{method_name}_{al_method}_K{K}"
               f"_seed{seed_k}_B{batch_B}_R{rounds}_holdout{int(holdout_frac*100)}")
        out_dir = os.path.join(out_root, "5_fed_baselines", tag)
        if os.path.exists(out_dir):
            continue
        run_one(base, updates, out_dir, tag, runner="tools/run_fel.py")

    # --- Gradient-based baselines (via run_baseline.py) ---
    baselines = [
        ("fedavg",  {"baseline_method": "fedavg",  "baseline_proximal_mu": 0.0}),
        ("fedprox", {"baseline_method": "fedprox", "baseline_proximal_mu": 0.01}),
        ("fednova", {"baseline_method": "fednova", "baseline_proximal_mu": 0.0}),
    ]

    for method_name, method_cfg in baselines:
        tag = (f"exp5_{method_name}_{al_method}_K{K}"
               f"_seed{seed_k}_B{batch_B}_R{rounds}_holdout{int(holdout_frac*100)}")
        out_dir = os.path.join(out_root, "5_fed_baselines", tag)
        if os.path.exists(out_dir):
            continue

        updates = {
            "seed_k": seed_k,
            "rounds": rounds,
            "al.method": al_method,
            "al.batch_B": batch_B,
            "holdout.test_frac": holdout_frac,
            "baseline_epochs": 5,
            "baseline_lr": 0.01,
            **method_cfg,
        }
        run_one(base, updates, out_dir, tag, runner="tools/run_baseline.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True, help="Path to your base YAML (keeps dataset csv paths).")
    ap.add_argument("--out-root", required=True, help="Where to store runs and derived configs.")
    ap.add_argument("--study", type=int, required=True, choices=[0,1,2,3,4,5,6,7,8,9,10])
    ap.add_argument("--clients", type=str, default="", help="Comma-separated client names if your config expects them (optional).")
    ap.add_argument("--scenario", type=str, default="baseline", help="Study 2/4 scenario name.")
    ap.add_argument("--label-noise", type=float, default=0.0)
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    if args.study == 0:
        study_eval(args.base_config, args.out_root)
    elif args.study == 1:
        study1(args.base_config, args.out_root)
    elif args.study == 2:
        study2(args.base_config, args.out_root)
    elif args.study == 3:
        study3(args.base_config, args.out_root)
    elif args.study == 4:
        study4(args.base_config, args.out_root)
    elif args.study == 5:
        study_scalability(args.base_config, args.out_root)
    elif args.study == 6:
        study_meta_ablation(args.base_config, args.out_root)
    elif args.study == 7:
        study_meta_hetero_rf(args.base_config, args.out_root)
    elif args.study == 8:
        study_meta_noise(args.base_config, args.out_root)
    elif args.study == 9:
        study_hetero_models(args.base_config, args.out_root)
    elif args.study == 10:
        study_fed_baselines(args.base_config, args.out_root)

if __name__ == "__main__":
    main()