import time, json
import numpy as np
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from .client import ClientState
from ..selection.k_coreset import kcenter_greedy
from ..selection.kmeans_seed import kmeans_seed
from ..meta.fed_ridge import FedRidgeMeta
from ..utils.data_io import load_clients_from_csvs, split_holdout
from ..utils.dirichlet_partition import partition_all_clients
from ..utils.calibration import expected_calibration_error, brier_score
from ..models.model_wrapper import build_model

class SimulationRunner:
    def __init__(self, client_pools, classes, rf_cfg, meta_cfg, seed_L, rounds, batch_size,
                 aug_K=0, alpha=0.0, per_class_min=1, refresh_every=1, al_method="qbc",
                 logger=None, out_dir: str|None=None, log_jsonl: str|None=None,
                 test_indices: dict|None=None,
                 per_client_rf_cfgs: dict|None=None,
                 per_client_noise: dict|None=None,
                 per_client_model_types: dict|None=None):
        """
        Parameters
        ----------
        test_indices : dict[str, list[int]] | None
            Per-client fixed test indices.
        per_client_rf_cfgs : dict[str, dict] | None
            Per-client RF hyperparameters. Keys are client IDs.
            If a client ID is not in the dict, falls back to rf_cfg.
        per_client_noise : dict[str, float] | None
            Per-client embedding noise std. Keys are client IDs.
        """
        self.client_pools = client_pools
        self.classes = classes
        self.rf_cfg = rf_cfg
        self.meta_cfg = meta_cfg
        self.seed_L = seed_L
        self.rounds = int(rounds)
        self.batch_size = int(batch_size)
        self.aug_K = int(aug_K)
        self.alpha = float(alpha)
        self.per_class_min = int(per_class_min)
        self.refresh_every = int(refresh_every)
        self.al_method = al_method
        self.logger = logger
        self.out_dir = Path(out_dir) if out_dir else None
        if self.out_dir: self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_jsonl = Path(log_jsonl) if log_jsonl else (self.out_dir/"sim_log.jsonl" if self.out_dir else None)
        self.clients = {}
        self.test_indices = test_indices or {}
        self.per_client_rf_cfgs = per_client_rf_cfgs or {}
        self.per_client_noise = per_client_noise or {}
        self.per_client_model_types = per_client_model_types or {}

    def _log(self, msg):
        print(msg) if self.logger is None else self.logger.info(msg)

    def run(self):
        # init clients
        self.clients = {}
        for cid, pool in self.client_pools.items():
            test_idx = self.test_indices.get(cid, [])
            # Pool originals exclude test indices
            pool_originals = pool["originals"]
            if test_idx:
                test_set = set(test_idx)
                pool_originals = [i for i in pool_originals if i not in test_set]

            # Build model: per-client model type or per-client RF config or global RF
            model_obj = None
            if cid in self.per_client_model_types:
                mtype = self.per_client_model_types[cid]
                rf_cfg_for_client = self.per_client_rf_cfgs.get(cid, self.rf_cfg)
                model_obj = build_model(mtype, self.classes, **rf_cfg_for_client)
                self._log(f"  [Init] {cid}: model={mtype}")

            c = ClientState(
                client_id=cid, classes=self.classes,
                emb=pool["emb"], y=pool["label"], axial=pool["axial"],
                originals=pool_originals, augs_by_axial=pool["augs_by_axial"],
                idx_by_axial=pool["idx_by_axial"],
                rf_cfg=self.per_client_rf_cfgs.get(cid, self.rf_cfg),
                al_method=self.al_method,
                train_aug_K=self.aug_K, tta_n=self.meta_cfg.get("tta_n", 0),
                meta=None,
                test_indices=test_idx,
                emb_noise_std=self.per_client_noise.get(cid, 0.0),
                noise_seed=hash(cid) % (2**31),
                model=model_obj,
            )
            c.L = sorted(self.seed_L.get(cid, []))
            c.U = sorted([i for i in c.originals if i not in set(c.L)])
            self.clients[cid] = c

        # set up META
        meta = FedRidgeMeta(classes=self.classes,
                            lam=self.meta_cfg.get("lambda", 1e-2),
                            oof_folds=self.meta_cfg.get("oof_folds", 5),
                            include_T=self.meta_cfg.get("include_T", False),
                            aug_mode=self.meta_cfg.get("aug_mode", "off"),
                            tta_n=self.meta_cfg.get("tta_n", 0),
                            feature_groups=self.meta_cfg.get("feature_groups", None))
        for c in self.clients.values():
            c.meta = meta

        t0 = time.time()
        for r in range(self.rounds):
            self._log(f"[FEL] Round {r+1}/{self.rounds}")

            # 1) Train local RFs
            t_train = time.time()
            for c in self.clients.values():
                mlbl = type(c.rf).__name__.replace('Wrapper','').replace('Model','')
                self._log(f"  [Train] {c.client_id}: |L|={len(c.L)} |U|={len(c.U)} (model={mlbl}, aug_K={self.aug_K})")
                c.fit_rf_with_aug(self.aug_K)
            t_train = time.time() - t_train

            # 2) META refresh
            t_meta = time.time()
            do_refresh = (r % self.refresh_every == 0) or (r == 0)
            if do_refresh:
                Sxx = Sxy = None
                sum_n  = 0
                sum_mu = None
                sum_m2 = None

                for c in self.clients.values():
                    if len(c.L) == 0:
                        continue
                    Xb = c.emb[c.L]; yb = c.y_pool[c.L]

                    if c._is_tree_model and hasattr(c.rf._clf, 'estimators_'):
                        # Tree-based: use per-tree OOF
                        trees = c.rf._clf.estimators_
                        group_ids = None
                        if meta.aug_mode == 'grouped':
                            group_ids = np.array([hash(str(c.axial[i])) % (10**9) for i in c.L], dtype=np.int64)
                        res = meta.client_oof_stats(
                            Xb, yb, trees, c.scaler,
                            group_ids=group_ids, aug_mode=meta.aug_mode,
                            forest_classes=list(c.rf._clf.classes_)
                        )
                    else:
                        # Non-tree model: re-fit OOF
                        res = meta.client_oof_stats_generic(
                            Xb, yb, c.rf, c.scaler
                        )
                    if res is None:
                        continue
                    sxx, sxy, mu_i, var_i, n_i = res

                    Sxx = sxx if Sxx is None else Sxx + sxx
                    Sxy = sxy if Sxy is None else Sxy + sxy

                    sum_n += n_i
                    if sum_mu is None:
                        sum_mu = n_i * mu_i
                        sum_m2 = n_i * (var_i + mu_i**2)
                    else:
                        sum_mu += n_i * mu_i
                        sum_m2 += n_i * (var_i + mu_i**2)

                if Sxx is not None and Sxy is not None and sum_n > 0:
                    mu_g = sum_mu / sum_n
                    var_g = (sum_m2 / sum_n) - mu_g**2
                    sigma_g = np.sqrt(np.maximum(var_g, 1e-12))
                    meta.server_solve(Sxx, Sxy, mu_g, sigma_g, with_bias=True)
                    self._log("  [Meta] ridge updated.")
                else:
                    self._log("  [Meta] skipped (insufficient labels).")

            t_meta = time.time() - t_meta

            # 3a) Evaluate on U (legacy — shrinking pool)
            t_eval = time.time()
            self._log("  [Eval] U (rest of pool):")
            agg_rf, agg_me = [], []
            per_client_metrics = {}
            for c in self.clients.values():
                m_rf = c.eval_split(split="U", use_meta=False)
                m_me = c.eval_split(split="U", use_meta=True)
                per_client_metrics[c.client_id] = {"rf": m_rf, "meta": m_me}
                def _fmt(x): return f"{x:.3f}" if x is not None else "na"
                mlbl = type(c.rf).__name__.replace('Wrapper','').replace('Model','')
                self._log(f"    - {c.client_id:10s} | {mlbl:4s} balAcc={_fmt(m_rf['balanced_acc'])} F1={_fmt(m_rf['macro_f1'])} AUC={_fmt(m_rf['ovr_auc'])} (n={m_rf['n']})")
                self._log(f"      {'':10s} | META balAcc={_fmt(m_me['balanced_acc'])} F1={_fmt(m_me['macro_f1'])} AUC={_fmt(m_me['ovr_auc'])} (n={m_me['n']})")
                agg_rf.append(m_rf); agg_me.append(m_me)
            def _avg(key, stats):
                vals = [s[key] for s in stats if s[key] is not None]
                return sum(vals)/len(vals) if vals else None
            avg_rf = {"balanced_acc": _avg("balanced_acc", agg_rf),
                      "macro_f1": _avg("macro_f1", agg_rf),
                      "ovr_auc": _avg("ovr_auc", agg_rf)}
            avg_me = {"balanced_acc": _avg("balanced_acc", agg_me),
                      "macro_f1": _avg("macro_f1", agg_me),
                      "ovr_auc": _avg("ovr_auc", agg_me)}

            # 3b) Evaluate on fixed held-out test set
            # Each sub-client evaluates independently on its own test set
            # (same as Exp 1 — treat sub-clients as real clients).
            agg_test_rf, agg_test_me = [], []
            per_client_test = {}
            # Collect probas across all clients for aggregate calibration
            all_ytrue_rf, all_proba_rf = [], []
            all_ytrue_me, all_proba_me = [], []
            has_test = any(len(c.test_indices) > 0 for c in self.clients.values())
            if has_test:
                self._log("  [Eval] Fixed test set:")
                for c in self.clients.values():
                    t_rf, yt_rf, p_rf = c.eval_fixed_test_with_proba(use_meta=False)
                    t_me, yt_me, p_me = c.eval_fixed_test_with_proba(use_meta=True)
                    per_client_test[c.client_id] = {"rf": t_rf, "meta": t_me}
                    mlbl = type(c.rf).__name__.replace('Wrapper','').replace('Model','')
                    self._log(f"    - {c.client_id:10s} | {mlbl:4s} F1={_fmt(t_rf['macro_f1'])} "
                              f"META F1={_fmt(t_me['macro_f1'])} (n={t_rf['n']})")
                    agg_test_rf.append(t_rf); agg_test_me.append(t_me)
                    if yt_rf is not None and p_rf is not None:
                        all_ytrue_rf.append(yt_rf); all_proba_rf.append(p_rf)
                    if yt_me is not None and p_me is not None:
                        all_ytrue_me.append(yt_me); all_proba_me.append(p_me)
                avg_test_rf = {"balanced_acc": _avg("balanced_acc", agg_test_rf),
                               "macro_f1": _avg("macro_f1", agg_test_rf),
                               "ovr_auc": _avg("ovr_auc", agg_test_rf)}
                avg_test_me = {"balanced_acc": _avg("balanced_acc", agg_test_me),
                               "macro_f1": _avg("macro_f1", agg_test_me),
                               "ovr_auc": _avg("ovr_auc", agg_test_me)}
                # Compute aggregate calibration metrics (pool all clients)
                cal_rf = {}
                cal_me = {}
                if all_ytrue_rf:
                    yt_cat = np.concatenate(all_ytrue_rf)
                    p_cat = np.vstack(all_proba_rf)
                    cal_rf = {"ece": expected_calibration_error(yt_cat, p_cat, self.classes),
                              "brier": brier_score(yt_cat, p_cat, self.classes)}
                if all_ytrue_me:
                    yt_cat = np.concatenate(all_ytrue_me)
                    p_cat = np.vstack(all_proba_me)
                    cal_me = {"ece": expected_calibration_error(yt_cat, p_cat, self.classes),
                              "brier": brier_score(yt_cat, p_cat, self.classes)}
                avg_test_rf.update(cal_rf)
                avg_test_me.update(cal_me)
                self._log(f"  [Eval][avg-test] RF F1={_fmt(avg_test_rf['macro_f1'])} "
                          f"ECE={cal_rf.get('ece','na'):.4f} Brier={cal_rf.get('brier','na'):.4f}"
                          if cal_rf else
                          f"  [Eval][avg-test] RF F1={_fmt(avg_test_rf['macro_f1'])}")
                self._log(f"  [Eval][avg-test] META F1={_fmt(avg_test_me['macro_f1'])} "
                          f"ECE={cal_me.get('ece','na'):.4f} Brier={cal_me.get('brier','na'):.4f}"
                          if cal_me else
                          f"  [Eval][avg-test] META F1={_fmt(avg_test_me['macro_f1'])}")
            else:
                avg_test_rf = avg_test_me = {"balanced_acc": None, "macro_f1": None, "ovr_auc": None}
                per_client_test = {}

            t_eval = time.time() - t_eval

            # 4) Acquisition
            t_acq = time.time()
            round_entry = {
                "round": r+1,
                "al_method": self.al_method,
                "clients": {},
                "avg": {"rf": avg_rf, "meta": avg_me},
                "timing_sec": {},
            }
            # NEW: include fixed-test metrics in log
            if has_test:
                round_entry["avg_test"] = {"rf": avg_test_rf, "meta": avg_test_me}

            for c in self.clients.values():
                acq = c.acquire(self.batch_size, alpha=self.alpha, per_class_min=self.per_class_min, method=self.al_method)
                picked_n = len(acq["picked_idx"])
                self._log(f"  [Acquire] {c.client_id}: picked {picked_n} → |L|={len(c.L)} |U|={len(c.U)}")
                client_entry = {
                    "model_type": type(c.rf).__name__.replace('Wrapper','').replace('Model',''),
                    "rf": per_client_metrics[c.client_id]["rf"],
                    "meta": per_client_metrics[c.client_id]["meta"],
                    "picked_per_class": acq.get("picked_counts", {}),
                    "picked_n": picked_n,
                    "train_rows": acq.get("train_rows_aug", len(c.L)),
                    "committee_size": acq.get("committee_size", 0),
                    "L_size": len(c.L),
                    "U_size": len(c.U),
                }
                # NEW: per-client test metrics
                if c.client_id in per_client_test:
                    client_entry["test_rf"] = per_client_test[c.client_id]["rf"]
                    client_entry["test_meta"] = per_client_test[c.client_id]["meta"]
                round_entry["clients"][c.client_id] = client_entry

            t_acq = time.time() - t_acq

            if self.log_jsonl:
                round_entry["timing_sec"] = {"train": t_train, "meta_refresh": t_meta, "eval": t_eval, "acquire": t_acq}
                with open(self.log_jsonl, "a") as f:
                    f.write(json.dumps(round_entry) + "\n")

        if self.log_jsonl:
            with open(self.log_jsonl, "a") as f:
                f.write(json.dumps({"summary": {"rounds": self.rounds, "wall_clock_sec": time.time()-t0}}) + "\n")

def run_sim(*, client_pools, classes, rf_cfg, meta_cfg, seed_L, rounds, batch_size,
            aug_K=0, alpha=0.0, per_class_min=1, refresh_every=1, al_method="qbc",
            logger=None, out_dir=None, test_indices=None,
            per_client_rf_cfgs=None, per_client_noise=None,
            per_client_model_types=None):
    runner = SimulationRunner(client_pools, classes, rf_cfg, meta_cfg, seed_L, rounds, batch_size,
                              aug_K, alpha, per_class_min, refresh_every, al_method, logger,
                              out_dir, log_jsonl=str(Path(out_dir)/"sim_log.jsonl") if out_dir else None,
                              test_indices=test_indices,
                              per_client_rf_cfgs=per_client_rf_cfgs,
                              per_client_noise=per_client_noise,
                              per_client_model_types=per_client_model_types)
    runner.run()

def run_sim_from_yaml(cfg: dict):
    import random, numpy as _np
    raw_rs = cfg.get("random_state", 42)
    rs = int(raw_rs)
    random.seed(rs); _np.random.seed(rs)
    try:
        import torch
        torch.manual_seed(rs); torch.cuda.manual_seed_all(rs)
    except Exception:
        pass

    rounds = int(cfg["rounds"]); seed_k = int(cfg.get("seed_k", 0))
    rf_cfg = cfg.get("rf", {})
    meta_cfg = cfg.get("meta", {})
    # Parse feature_groups: YAML stores as comma-separated string, code needs list
    fg = meta_cfg.get("feature_groups", None)
    if isinstance(fg, str):
        meta_cfg["feature_groups"] = [g.strip() for g in fg.split(",") if g.strip()]
    al = cfg.get("al", {})
    batch_size = int(al.get("batch_B", 32))
    per_class_min = int(al.get("per_class_min", 1))
    al_method = al.get("method", "qbc")
    augment = cfg.get("augment", {}); aug_K = int(augment.get("train_n_per_sample", 0))

    data = cfg["data"]
    label_col = data.get("label_col", "ivd_level")
    id_col    = data.get("id_col", "id")
    image_col = data.get("image_col", "axial_path")
    emb_col   = data.get("emb_col", "emb")

    csv_map = {}
    if "rsna_csv" in data: csv_map["RSNA"] = data["rsna_csv"]
    if "mendeley_csv" in data: csv_map["MENDELEY"] = data["mendeley_csv"]
    pools = load_clients_from_csvs(csv_map, label_col, id_col, emb_col, axial_col=image_col)

    labels = []
    for p in pools.values(): labels.append(p["label"][p["originals"]])
    classes = sorted(np.unique(np.concatenate(labels)).tolist())

    # --- NEW: optional holdout split ---
    # Holdout is done on the REAL clients BEFORE any Dirichlet partitioning,
    # so the test set remains fixed regardless of partition config.
    holdout_cfg = cfg.get("holdout", {})
    holdout_frac = float(holdout_cfg.get("test_frac", 0.0))
    real_test_indices = {}  # keyed by real client id
    if holdout_frac > 0:
        for cid, pool in pools.items():
            pool_orig, test_orig = split_holdout(pool, test_frac=holdout_frac, seed=rs)
            real_test_indices[cid] = test_orig
            # Replace pool originals so seeding only uses pool portion
            pools[cid]["originals"] = pool_orig

    # --- NEW: optional partition (Experiment 2) ---
    # Splits real clients into C simulated sub-clients.
    # "dirichlet": non-IID class distributions controlled by alpha.
    # "uniform":   IID random split (alpha=∞ control).
    # Must happen AFTER holdout split but BEFORE seeding.
    partition_cfg = cfg.get("partition", {})
    partition_method = partition_cfg.get("method", None)
    if partition_method in ("dirichlet", "uniform"):
        n_clients = int(partition_cfg["n_clients"])
        if partition_method == "uniform":
            # Uniform = Dirichlet with very large alpha (effectively IID)
            dir_alpha = 1e6
            print(f"[Partition] Uniform (IID) split into {n_clients} simulated clients")
        else:
            dir_alpha = float(partition_cfg["alpha"])
            print(f"[Partition] Dirichlet split into {n_clients} simulated clients "
                  f"(alpha={dir_alpha})")
        pools = partition_all_clients(pools, n_clients, dir_alpha, seed=rs)
        for cid, pool in pools.items():
            n_orig = len(pool["originals"])
            cls_dist = {}
            for idx in pool["originals"]:
                c = pool["label"][idx]
                cls_dist[c] = cls_dist.get(c, 0) + 1
            print(f"  {cid}: {n_orig} originals, class dist={cls_dist}")

    # Map test indices to simulated client ids.
    # After partitioning, sub-client keys are "{REAL_ID}_{sub_idx}".
    # Each sub-client should evaluate on its parent's test set.
    test_indices = {}
    if real_test_indices:
        for cid in pools.keys():
            # Find the real parent client id
            for real_id in real_test_indices:
                if cid == real_id or cid.startswith(f"{real_id}_"):
                    test_indices[cid] = real_test_indices[real_id]
                    break

    # --- Seeding (on pool originals only) ---
    # With many simulated clients, per-client pool may be small.
    # Scale seed_k down to avoid seeding more than a quarter of the pool,
    # leaving ≥75% for active learning acquisition.
    seed_cfg = cfg.get("seeding", {})
    seed_method = seed_cfg.get("method", "k_coreset")
    seed_L = {}
    n_classes = len(classes)
    for cid, pool in pools.items():
        pool_size = len(pool["originals"])
        # Adaptive seed budget: at most 25% of pool, at least n_classes
        effective_seed_k = max(n_classes, min(seed_k, pool_size // 4))
        X = pool["emb"][pool["originals"]]
        if seed_method == "kmeans":
            idx_rel = kmeans_seed(X, k=min(effective_seed_k, X.shape[0]), random_state=rs)
        else:
            idx_rel = kcenter_greedy(X, k=min(effective_seed_k, X.shape[0]), seed=rs)
        seed_L[cid] = [int(pool["originals"][i]) for i in idx_rel]

    out_dir = cfg.get("logging", {}).get("out_dir")

    # Scale batch size if partitioned: keep total labels/round constant
    # by dividing B across simulated clients
    effective_batch = batch_size
    if partition_method in ("dirichlet", "uniform"):
        n_real = len(real_test_indices) if real_test_indices else 2
        n_sim = len(pools)
        # Each simulated client gets B * (n_real / n_sim) labels per round
        # so total labels/round stays at B * n_real
        effective_batch = max(1, int(round(batch_size * n_real / n_sim)))
        print(f"[Partition] Scaled batch_size: {batch_size} → {effective_batch} per sub-client "
              f"(total/round ≈ {effective_batch * n_sim})")

    # --- Exp 3b/3c: per-client heterogeneous RF configs and embedding noise ---
    import json as _json
    per_client_rf_cfgs = {}
    per_client_noise = {}

    # heterogeneous_rf: JSON list of rf config dicts, assigned round-robin
    hetero_rf_raw = cfg.get("heterogeneous_rf", None)
    if hetero_rf_raw:
        if isinstance(hetero_rf_raw, str):
            rf_config_list = _json.loads(hetero_rf_raw)
        else:
            rf_config_list = list(hetero_rf_raw)
        client_ids = sorted(pools.keys())
        for i, cid in enumerate(client_ids):
            base = dict(rf_cfg)  # copy global defaults
            base.update(rf_config_list[i % len(rf_config_list)])
            per_client_rf_cfgs[cid] = base
        print(f"[Heterogeneous RF] {len(rf_config_list)} configs → {len(client_ids)} clients")
        for cid, rc in per_client_rf_cfgs.items():
            print(f"  {cid}: n_est={rc.get('n_estimators','?')} max_depth={rc.get('max_depth','None')}")

    # emb_noise_stds: JSON list of floats, assigned round-robin
    noise_raw = cfg.get("emb_noise_stds", None)
    if noise_raw:
        if isinstance(noise_raw, str):
            noise_list = _json.loads(noise_raw)
        else:
            noise_list = list(noise_raw)
        client_ids = sorted(pools.keys())
        for i, cid in enumerate(client_ids):
            per_client_noise[cid] = float(noise_list[i % len(noise_list)])
        print(f"[Emb Noise] {len(noise_list)} stds → {len(client_ids)} clients")
        for cid, ns in per_client_noise.items():
            print(f"  {cid}: noise_std={ns:.4f}")

    # --- Exp 3d: per-client model types (mixed architecture federation) ---
    per_client_model_types = {}
    model_types_raw = cfg.get("client_model_types", None)
    if model_types_raw:
        if isinstance(model_types_raw, str):
            model_list = _json.loads(model_types_raw)
        else:
            model_list = list(model_types_raw)
        client_ids = sorted(pools.keys())
        for i, cid in enumerate(client_ids):
            per_client_model_types[cid] = model_list[i % len(model_list)]
        print(f"[Model Types] {len(model_list)} types → {len(client_ids)} clients")
        for cid, mt in per_client_model_types.items():
            print(f"  {cid}: {mt}")

    run_sim(client_pools=pools, classes=classes, rf_cfg=rf_cfg, meta_cfg=meta_cfg,
            seed_L=seed_L, rounds=rounds, batch_size=effective_batch, aug_K=aug_K, alpha=0.0,
            per_class_min=per_class_min, refresh_every=int(meta_cfg.get("refresh_every",1)),
            al_method=al_method, out_dir=out_dir,
            test_indices=test_indices,
            per_client_rf_cfgs=per_client_rf_cfgs,
            per_client_noise=per_client_noise,
            per_client_model_types=per_client_model_types)
