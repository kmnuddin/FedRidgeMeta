#!/usr/bin/env python3
"""Run a federated baseline (FedAvg / FedProx / FedNova) from a YAML config."""
import sys, argparse, yaml

sys.path.insert(0, "src")
from fel_ivd_federated.models.fed_baselines import run_baseline_from_yaml

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_baseline_from_yaml(cfg)
