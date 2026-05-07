"""
centralized_runner.py — Centralized (non-federated) upper-bound baseline.

Pools all client embeddings and labels into a single dataset, then runs
the same AL loop (seeding → train RF → optional ridge meta → acquire →
repeat) on the pooled data.  Evaluates on the same fixed held-out test
set used by the federated runs, enabling a direct "gap to oracle"
comparison.

Usage:
    Called from fel_sweeps.py study_eval or via run_centralized_from_yaml().
"""
import time, json, math
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from collections import Counter

from ..models.rf import RFWrapper
from ..meta.fed_ridge import FedRidgeMeta
from ..meta.summary_features import summary_features_from_tree_probs, entropy
from ..meta.fed_ridge import _pad_tree_proba
from ..selection.k_coreset import kcenter_greedy
from ..selection.kmeans_seed import kmeans_seed
from ..utils.data_io import load_clients_from_csvs, split_holdout
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


class CentralizedRunner:
    """
    Simulates a centralized (non-federated) oracle baseline.

    All client data is pooled into a single embedding matrix.  A single RF
    is trained on the labeled set, an optional ridge meta-learner is fitted
    on OOF meta-features (no federation — just a single-client solve), and
    acquisition uses the same AL strategies as the federated loop.
    """

    def __init__(self, emb, y, axial, originals, augs_by_axial,
                 classes, rf_cfg, meta_cfg,
                 seed_L, rounds, batch_size,
                 al_method="least_confident",
                 aug_K=0, per_class_min=1,
                 test_indices=None,
                 out_dir=None, logger=None):
        self.emb = emb
        self.y = np.asarray(y)
        self.axial = np.asarray(axial)
        self.originals = sorted(originals)
        self.augs_by_axial = augs_by_axial
        self.classes = classes
        self.rf_cfg = rf_cfg
        self.meta_cfg = meta_cfg
        self.seed_L = sorted(seed_L)
        self.rounds = int(rounds)
        self.batch_size = int(batch_size)
        self.al_method = al_method
        self.aug_K = int(aug_K)
        self.per_class_min = int(per_class_min)
        self.test_indices = sorted(test_indices) if test_indices else []
        self.out_dir = Path(out_dir) if out_dir else None
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_jsonl = self.out_dir / "sim_log.jsonl" if self.out_dir else None
        self.logger = logger

        # state
        self.L = []
        self.U = []
        self.scaler = StandardScaler()
        self.rf = RFWrapper(**rf_cfg)
        self.rf.all_classes = list(classes)
        self.meta = None

    def _log(self, msg):
        print(msg) if self.logger is None else self.logger.info(msg)

    # ----- training -----

    def _fit_rf(self):
        if not self.L:
            return
        Z_L = self.emb[self.L]
        self.scaler.fit(Z_L)

        X_list = [Z_L]
        y_list = [self.y[self.L]]
        for idx in self.L:
            ax = self.axial[idx]
            augs = self.augs_by_axial.get(ax, [])[:min(self.aug_K, len(self.augs_by_axial.get(ax, [])))]
            if augs:
                X_list.append(self.emb[augs])
                y_list.append(self.y[augs])
        Xtr = np.vstack(X_list)
        ytr = np.concatenate(y_list)
        self.rf.fit(self.scaler.transform(Xtr), ytr)

    def _fit_meta(self):
        """Single-pool ridge (equivalent to 1-client federation)."""
        if not self.L or self.meta is None:
            return
        Xb = self.emb[self.L]
        yb = self.y[self.L]
        trees = self.rf._clf.estimators_

        res = self.meta.client_oof_stats(Xb, yb, trees, self.scaler,
                                         forest_classes=list(self.rf._clf.classes_))
        if res is None:
            return
        sxx, sxy, mu, var, n = res
        sigma = np.sqrt(np.maximum(var, 1e-12))
        self.meta.server_solve(sxx, sxy, mu, sigma, with_bias=True)

    # ----- evaluation -----

    def _eval(self, idx, use_meta=False):
        if not idx:
            return {"balanced_acc": None, "macro_f1": None, "ovr_auc": None, "n": 0}
        Z = self.scaler.transform(self.emb[idx])
        if use_meta and self.meta is not None and self.meta.W is not None:
            trees = self.rf._clf.estimators_
            fc = list(self.rf._clf.classes_)
            tp = [_pad_tree_proba(t, Z, self.classes, forest_classes=fc) for t in trees]
            Xf, _ = summary_features_from_tree_probs(tp, self.classes,
                                                        feature_groups=self.meta.feature_groups)
            P = self.meta.predict_proba(Xf)
        else:
            P = self.rf.predict_proba(Z)

        y_true = self.y[idx]
        y_pred = np.array([self.classes[i] for i in P.argmax(axis=1)])
        out = {"n": len(idx)}
        try: out["balanced_acc"] = float(balanced_accuracy_score(y_true, y_pred))
        except: out["balanced_acc"] = None
        try: out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
        except: out["macro_f1"] = None
        try: out["ovr_auc"] = float(roc_auc_score(y_true, P, multi_class="ovr", labels=self.classes))
        except: out["ovr_auc"] = None
        return out

    # ----- acquisition -----

    def _acquire(self, B):
        if B <= 0 or not self.U:
            return []
        ZU = self.scaler.transform(self.emb[self.U])
        trees = self.rf._clf.estimators_
        fc = list(self.rf._clf.classes_)
        tree_P = np.stack([_pad_tree_proba(t, ZU, self.classes, forest_classes=fc) for t in trees], axis=0)
        mean_p = tree_P.mean(axis=0)

        if self.al_method == "entropy":
            scores = entropy(mean_p)
        elif self.al_method == "margin":
            part = np.partition(-mean_p, 2, axis=1)
            scores = 1.0 - (-part[:, 0] - (-part[:, 1]))
        elif self.al_method == "least_confident":
            scores = 1.0 - mean_p.max(axis=1)
        elif self.al_method == "bald":
            scores = entropy(mean_p) - entropy(tree_P).mean(axis=0)
        else:  # qbc / JS
            eps = 1e-12
            m = np.clip(mean_p[None, :, :], eps, 1.0)
            P = np.clip(tree_P, eps, 1.0)
            M = 0.5 * (P + m)
            js = 0.5 * ((P * np.log(P / M)).sum(axis=2) + (m * np.log(m / M)).sum(axis=2))
            scores = js.mean(axis=0)

        preds = mean_p.argmax(axis=1)
        order = np.argsort(-scores, kind="mergesort")
        taken = set()
        per_class_taken = {c: 0 for c in self.classes}
        for i in order:
            c = self.classes[preds[i]]
            if per_class_taken[c] < self.per_class_min:
                taken.add(i); per_class_taken[c] += 1
            if len(taken) >= min(B, len(self.U)):
                break
        for i in order:
            if len(taken) >= min(B, len(self.U)):
                break
            if i not in taken:
                taken.add(i)

        picked_idx = [self.U[i] for i in sorted(taken)]
        return picked_idx

    # ----- main loop -----

    def run(self):
        # initialise L / U
        self.L = list(self.seed_L)
        test_set = set(self.test_indices)
        self.U = sorted([i for i in self.originals if i not in set(self.L) and i not in test_set])

        # optional meta
        if self.meta_cfg.get("aug_mode", "off") != "disabled":
            self.meta = FedRidgeMeta(
                classes=self.classes,
                lam=self.meta_cfg.get("lambda", 1e-2),
                oof_folds=self.meta_cfg.get("oof_folds", 5),
                include_T=self.meta_cfg.get("include_T", False),
                aug_mode=self.meta_cfg.get("aug_mode", "off"),
                tta_n=self.meta_cfg.get("tta_n", 0),
                feature_groups=self.meta_cfg.get("feature_groups", None),
            )

        t0 = time.time()
        for r in range(self.rounds):
            self._log(f"[CENTRAL] Round {r+1}/{self.rounds}  |L|={len(self.L)} |U|={len(self.U)}")

            t_train = time.time()
            self._fit_rf()
            t_train = time.time() - t_train

            t_meta = time.time()
            self._fit_meta()
            t_meta = time.time() - t_meta

            t_eval = time.time()
            m_u_rf   = self._eval(self.U, use_meta=False)
            m_u_meta = self._eval(self.U, use_meta=True)
            m_t_rf   = self._eval(self.test_indices, use_meta=False)
            m_t_meta = self._eval(self.test_indices, use_meta=True)
            t_eval = time.time() - t_eval

            def _f(x): return f"{x:.3f}" if x is not None else "na"
            self._log(f"  [U]    RF F1={_f(m_u_rf['macro_f1'])} META F1={_f(m_u_meta['macro_f1'])}")
            self._log(f"  [Test] RF F1={_f(m_t_rf['macro_f1'])} META F1={_f(m_t_meta['macro_f1'])}")

            t_acq = time.time()
            picked = self._acquire(self.batch_size)
            self.L = sorted(set(self.L) | set(picked))
            self.U = sorted([i for i in self.U if i not in set(self.L)])
            t_acq = time.time() - t_acq

            picked_counts = {}
            for idx in picked:
                c = self.y[idx]
                picked_counts[c] = picked_counts.get(c, 0) + 1

            entry = {
                "round": r + 1,
                "mode": "centralized",
                "al_method": self.al_method,
                "avg": {"rf": m_u_rf, "meta": m_u_meta},
                "avg_test": {"rf": m_t_rf, "meta": m_t_meta},
                "picked_per_class": picked_counts,
                "picked_n": len(picked),
                "L_size": len(self.L),
                "U_size": len(self.U),
                "timing_sec": {"train": t_train, "meta_refresh": t_meta, "eval": t_eval, "acquire": t_acq},
            }
            if self.log_jsonl:
                with open(self.log_jsonl, "a") as f:
                    f.write(json.dumps(entry) + "\n")

        if self.log_jsonl:
            with open(self.log_jsonl, "a") as f:
                f.write(json.dumps({"summary": {"mode": "centralized", "rounds": self.rounds,
                                                 "wall_clock_sec": time.time() - t0}}) + "\n")
        self._log(f"[CENTRAL] Done. Wall clock: {time.time()-t0:.1f}s")


def run_centralized_from_yaml(cfg: dict):
    """
    Entry point that mirrors run_sim_from_yaml but pools all clients
    and runs the centralized baseline.  Expects the same YAML schema
    with an added `holdout.test_frac` field.
    """
    import random
    rs = int(cfg.get("random_state", 42))
    random.seed(rs); np.random.seed(rs)

    rounds = int(cfg["rounds"])
    seed_k = int(cfg.get("seed_k", 0))
    rf_cfg = cfg.get("rf", {})
    meta_cfg = cfg.get("meta", {})
    al = cfg.get("al", {})
    batch_size = int(al.get("batch_B", 32))
    per_class_min = int(al.get("per_class_min", 1))
    al_method = al.get("method", "qbc")
    augment = cfg.get("augment", {})
    aug_K = int(augment.get("train_n_per_sample", 0))

    data = cfg["data"]
    label_col = data.get("label_col", "ivd_level")
    id_col = data.get("id_col", "id")
    image_col = data.get("image_col", "axial_path")
    emb_col = data.get("emb_col", "emb")

    csv_map = {}
    if "rsna_csv" in data: csv_map["RSNA"] = data["rsna_csv"]
    if "mendeley_csv" in data: csv_map["MENDELEY"] = data["mendeley_csv"]
    pools = load_clients_from_csvs(csv_map, label_col, id_col, emb_col, axial_col=image_col)

    # --- holdout split (must use same seed / fraction as federated runs) ---
    holdout_cfg = cfg.get("holdout", {})
    holdout_frac = float(holdout_cfg.get("test_frac", 0.30))

    # Pool everything
    all_emb, all_y, all_axial = [], [], []
    all_originals = []
    all_augs: dict = {}
    all_test = []
    offset = 0

    for cid, pool in sorted(pools.items()):
        n = pool["emb"].shape[0]

        # split holdout per client (same split as federated)
        pool_orig, test_orig = split_holdout(pool, test_frac=holdout_frac, seed=rs)

        all_emb.append(pool["emb"])
        all_y.extend(pool["label"].tolist())
        all_axial.extend([f"{cid}__{a}" for a in pool["axial"]])  # prefix to avoid collisions

        for idx in pool_orig:
            all_originals.append(idx + offset)
        for idx in test_orig:
            all_test.append(idx + offset)

        for ax, aug_list in pool["augs_by_axial"].items():
            all_augs[f"{cid}__{ax}"] = [a + offset for a in aug_list]

        offset += n

    emb = np.vstack(all_emb)
    y = np.array(all_y)
    axial = all_axial
    classes = sorted(np.unique(y[all_originals + all_test]).tolist())

    # --- seeding on pooled originals ---
    seed_cfg = cfg.get("seeding", {})
    seed_method = seed_cfg.get("method", "k_coreset")
    X_pool = emb[all_originals]
    total_seed = min(seed_k * len(pools), len(all_originals))  # same total budget
    if seed_method == "kmeans":
        idx_rel = kmeans_seed(X_pool, k=total_seed, random_state=rs)
    else:
        idx_rel = kcenter_greedy(X_pool, k=total_seed, seed=rs)
    seed_L = [all_originals[i] for i in idx_rel]

    out_dir = cfg.get("logging", {}).get("out_dir")

    runner = CentralizedRunner(
        emb=emb, y=y, axial=axial, originals=all_originals,
        augs_by_axial=all_augs, classes=classes,
        rf_cfg=rf_cfg, meta_cfg=meta_cfg,
        seed_L=seed_L, rounds=rounds, batch_size=batch_size * len(pools),  # same total per round
        al_method=al_method, aug_K=aug_K, per_class_min=per_class_min,
        test_indices=all_test, out_dir=out_dir,
    )
    runner.run()
