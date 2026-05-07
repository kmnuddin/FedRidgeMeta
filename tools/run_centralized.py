#!/usr/bin/env python3
"""
run_centralized.py — Entry point for the centralized (non-federated)
upper-bound baseline.  Same CLI as run_fel.py.

Usage:
    PYTHONPATH=src python tools/run_centralized.py --config <derived.yaml>
"""
import argparse, yaml, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fel_ivd_federated.loop.centralized_runner import run_centralized_from_yaml

def main():
    ap = argparse.ArgumentParser("Centralized upper-bound baseline")
    ap.add_argument("--config", required=True, help="Path to FEL YAML config")
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    run_centralized_from_yaml(cfg)

if __name__ == "__main__":
    sys.exit(main())
