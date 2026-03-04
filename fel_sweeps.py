#!/usr/bin/env python3
"""
fel_sweeps.py — minimal-edit sweep runner for your FEL repo.

Usage examples (from project root):
  python fel_sweeps.py --base-config configs/example.yaml --out-root runs/fel_sweeps --study 1
  python fel_sweeps.py --base-config configs/example.yaml --out-root runs/fel_sweeps --study 2 --clients RSNA,MENDELEY
  python fel_sweeps.py --base-config configs/example.yaml --out-root runs/fel_sweeps --study 4 --scenario skew_3to1 --label-noise 0.05

The script:
- Loads your base YAML (keeps all dataset paths as-is).
- Writes derived YAMLs per run (seed_k, AL method, K, META, RF, etc.).
- Computes rounds from desired label budgets (seed_k + rounds*B).
- Calls: PYTHONPATH=src python tools/run_fel.py --config <derived.yaml>
- Stores each run under out-root/<study_id>/<short_name>/ with its YAML and JSONL logs.

For study 4 (Non-IID & label noise), see SCENARIOS.md for how to prep scenario CSVs and (optionally) enable a 10-line label-noise hook.
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
        # support dotted keys like 'al.method'
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

def run_one(base_cfg_path, cfg_updates, out_dir, tag):
    cfg = load_yaml(base_cfg_path)
    cfg = set_cfg(cfg, cfg_updates)
    cfg = ensure_logging_dir(cfg, out_dir)
    # Write derived yaml
    ypath = Path(out_dir) / f"derived_{tag}.yaml"
    dump_yaml(cfg, ypath)
    # Launch
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    cmd = ["python", "tools/run_fel.py", "--config", str(ypath)]
    print(">>> RUN:", " ".join(cmd))
    print(">>> OUT:", out_dir)
    subprocess.run(cmd, env=env, check=True)


def study1(base, out_root, n_estimators=(100, 500, 1000, 1500), max_features=("sqrt","log2"), batch_B=200, seed_k=200, add_labels=12000, K=0):
    """
    Committee size & diversity — fix AL sampler to qbc; TTA off; vary RF size/features.
    """
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
    """
    Non-IID stress & label noise. Prepare scenario-specific CSVs first (see SCENARIOS.md).
    To toggle label noise, see the optional hook in SCENARIOS.md.
    """
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
            # Optionally point to scenario CSVs if your base config supports it:
            # "data.csvs.RSNA": f"data/scenarios/{scenario}/RSNA.csv",
            # "data.csvs.MENDELEY": f"data/scenarios/{scenario}/MENDELEY.csv",
        }
        run_one(base, updates, out_dir, tag)

def study3(base, out_root, seed_k=200, add_labels=12000, batch_B_list=(50,100,200), K=0, lambdas=(1e-3,1e-2,1e-1,1,1e1,1e2,1e3)):
    """
    Small add-ons: seeding method, batch size sensitivity, ridge lambda stability.
    """
    rounds_map = {B: derive_rounds(seed_k, add_labels, B) for B in batch_B_list}

    # A) seeding k_coreset vs kmeans
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

    # B) Batch size sensitivity
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

    # C) Ridge lambda stability
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

def study4(
    base,
    out_root,
    seed_k=200,
    add_labels=12000,
    batch_B=200,
    methods=("bald","qbc","entropy","margin","least_confident"),
    n_aug=(0,8,16,24,32),
    meta_modes=("off","tta"),
    tta_grid=(4,8,16,24,32),
    refresh_every=1,
):
    """
    Study 4: Effect of AL sampler × augmentation (K) × META aug mode.
    If META mode != 'off', sweep tta_n over tta_grid.
    """
    rounds = derive_rounds(seed_k, add_labels, batch_B)

    for m in methods:
        for K in n_aug:
            
            for mode in meta_modes:
                # If META mode is 'off', force tta_n=0; otherwise sweep provided tta_grid.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True, help="Path to your base YAML (keeps dataset csv paths).")
    ap.add_argument("--out-root", required=True, help="Where to store runs and derived configs.")
    ap.add_argument("--study", type=int, required=True, choices=[1,2,3,4])
    # Optional toggles
    ap.add_argument("--clients", type=str, default="", help="Comma-separated client names if your config expects them (optional).")
    ap.add_argument("--scenario", type=str, default="baseline", help="Study 4 scenario name (folder under data/scenarios).")
    ap.add_argument("--label-noise", type=float, default=0.0, help="If you implemented the hook, pass 0.05 or 0.10 for study 4.")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    if args.study == 1:
        study1(args.base_config, args.out_root)
    elif args.study == 2:
        study2(args.base_config, args.out_root)
    elif args.study == 3:
        study3(args.base_config, args.out_root)
    elif args.study == 4:
        study4(args.base_config, args.out_root)

if __name__ == "__main__":
    main()
