import argparse, yaml, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fel_ivd_federated.loop.sim_runner import run_sim_from_yaml

def main():
    ap = argparse.ArgumentParser("FEL simulator")
    ap.add_argument("--config", required=True, help="Path to FEL YAML config")
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    run_sim_from_yaml(cfg)

if __name__ == "__main__":
    sys.exit(main())
