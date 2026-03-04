# FEL for Lumbar IVD — Federated QBC + META (embeddings-only)

- Pools and AL operate on **originals only** keyed by `axial_path`.
- RF training can include up to `augment.train_n_per_sample` augs per labeled original.
- META: `off` | `tta` | `grouped` (see `configs/example.yaml`).

## Run
PYTHONPATH=src python tools/run_fel.py --config configs/example.yaml

## CSV schema
- `axial_path`, `ivd_level`, `emb`, optional `is_aug`, `aug_idx`.
