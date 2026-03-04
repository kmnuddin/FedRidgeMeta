import time, json
import numpy as np
from pathlib import Path
from .client import ClientState
from ..selection.k_coreset import kcenter_greedy
from ..selection.kmeans_seed import kmeans_seed
from ..meta.fed_ridge import FedRidgeMeta
from ..utils.data_io import load_clients_from_csvs

class SimulationRunner:
    def __init__(self, client_pools, classes, rf_cfg, meta_cfg, seed_L, rounds, batch_size,
                 aug_K=0, alpha=0.0, per_class_min=1, refresh_every=1, al_method="qbc",
                 logger=None, out_dir: str|None=None, log_jsonl: str|None=None):
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

    def _log(self, msg):
        print(msg) if self.logger is None else self.logger.info(msg)

    def run(self):
        # init clients
        self.clients = {}
        for cid, pool in self.client_pools.items():
            c = ClientState(
                client_id=cid, classes=self.classes,
                emb=pool["emb"], y=pool["label"], axial=pool["axial"],
                originals=pool["originals"], augs_by_axial=pool["augs_by_axial"],
                idx_by_axial=pool["idx_by_axial"],
                rf_cfg=self.rf_cfg, al_method=self.al_method,
                train_aug_K=self.aug_K, tta_n=self.meta_cfg.get("tta_n", 0),
                meta=None
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
                            tta_n=self.meta_cfg.get("tta_n", 0))
        for c in self.clients.values():
            c.meta = meta

        t0 = time.time()
        for r in range(self.rounds):
            self._log(f"[FEL] Round {r+1}/{self.rounds}")

            # 1) Train local RFs
            t_train = time.time()
            for c in self.clients.values():
                self._log(f"  [Train] {c.client_id}: |L|={len(c.L)} |U|={len(c.U)} (aug_K={self.aug_K})")
                c.fit_rf_with_aug(self.aug_K)
            t_train = time.time() - t_train

            # 2) META refresh
            t_meta = time.time()
            do_refresh = (r % self.refresh_every == 0) or (r == 0)
            if do_refresh:
                Sxx = Sxy = None
            
                # accumulators for pooled raw-feature moments across clients (for inference z-score)
                sum_n  = 0                 # total OOF rows
                sum_mu = None              # ∑ n_i * mu_i   (mu_i is [1,D])
                sum_m2 = None              # ∑ n_i * (var_i + mu_i^2)
            
                for c in self.clients.values():
                    if len(c.L) == 0: 
                        continue
                    Xb = c.emb[c.L]; yb = c.y_pool[c.L]
                    trees = c.rf._clf.estimators_
            
                    group_ids = None
                    if meta.aug_mode == 'grouped':
                        group_ids = np.array([hash(str(c.axial[i])) % (10**9) for i in c.L], dtype=np.int64)
            
                    # UPDATED: client returns (Sxx, Sxy, mu_local, var_local, n_rows)
                    res = meta.client_oof_stats(
                        Xb, yb, trees, c.scaler,
                        group_ids=group_ids, aug_mode=meta.aug_mode
                    )
                    if res is None:
                        continue
                    sxx, sxy, mu_i, var_i, n_i = res
            
                    # accumulate ridge normal equations (already on z-scored + bias features)
                    Sxx = sxx if Sxx is None else Sxx + sxx
                    Sxy = sxy if Sxy is None else Sxy + sxy
            
                    # accumulate pooled raw-feature moments for global μ/σ
                    sum_n += n_i
                    if sum_mu is None:
                        sum_mu = n_i * mu_i
                        sum_m2 = n_i * (var_i + mu_i**2)
                    else:
                        sum_mu += n_i * mu_i
                        sum_m2 += n_i * (var_i + mu_i**2)
            
                if Sxx is not None and Sxy is not None and sum_n > 0:
                    # finalize global μ/σ for inference-time standardization
                    mu_g = sum_mu / sum_n
                    var_g = (sum_m2 / sum_n) - mu_g**2
                    sigma_g = np.sqrt(np.maximum(var_g, 1e-12))
            
                    # UPDATED: pass μ/σ and enable bias so inference matches training
                    meta.server_solve(Sxx, Sxy, mu_g, sigma_g, with_bias=True)
                    self._log("  [Meta] ridge updated.")
                else:
                    self._log("  [Meta] skipped (insufficient labels).")


            t_meta = time.time() - t_meta

            # 3) Evaluate on U
            t_eval = time.time()
            self._log("  [Eval] U (rest of pool):")
            agg_rf, agg_me = [], []
            per_client_metrics = {}
            for c in self.clients.values():
                m_rf = c.eval_split(split="U", use_meta=False)
                m_me = c.eval_split(split="U", use_meta=True)
                per_client_metrics[c.client_id] = {"rf": m_rf, "meta": m_me}
                def _fmt(x): return f"{x:.3f}" if x is not None else "na"
                self._log(f"    - {c.client_id:10s} | RF   balAcc={_fmt(m_rf['balanced_acc'])} F1={_fmt(m_rf['macro_f1'])} AUC={_fmt(m_rf['ovr_auc'])} (n={m_rf['n']})")
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
            self._log(f"  [Eval][avg] RF   balAcc={avg_rf['balanced_acc'] if avg_rf['balanced_acc'] is not None else 'na'} F1={avg_rf['macro_f1'] if avg_rf['macro_f1'] is not None else 'na'} AUC={avg_rf['ovr_auc'] if avg_rf['ovr_auc'] is not None else 'na'}")
            self._log(f"  [Eval][avg] META balAcc={avg_me['balanced_acc'] if avg_me['balanced_acc'] is not None else 'na'} F1={avg_me['macro_f1'] if avg_me['macro_f1'] is not None else 'na'} AUC={avg_me['ovr_auc'] if avg_me['ovr_auc'] is not None else 'na'}")
            t_eval = time.time() - t_eval

            # 4) Acquisition
            t_acq = time.time()
            round_entry = {
                "round": r+1,
                "al_method": self.al_method,
                "clients": {},
                "avg": {"rf": avg_rf, "meta": avg_me},
                "timing_sec": {}
            }
            for c in self.clients.values():
                acq = c.acquire(self.batch_size, alpha=self.alpha, per_class_min=self.per_class_min, method=self.al_method)
                picked_n = len(acq["picked_idx"])
                self._log(f"  [Acquire] {c.client_id}: picked {picked_n} → |L|={len(c.L)} |U|={len(c.U)}")
                round_entry["clients"][c.client_id] = {
                    "rf": per_client_metrics[c.client_id]["rf"],
                    "meta": per_client_metrics[c.client_id]["meta"],
                    "picked_per_class": acq.get("picked_counts", {}),
                    "picked_n": picked_n,
                    "train_rows": acq.get("train_rows_aug", len(c.L)),
                    "committee_size": acq.get("committee_size", 0),
                    "L_size": len(c.L),
                    "U_size": len(c.U),
                }
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
            logger=None, out_dir=None):
    runner = SimulationRunner(client_pools, classes, rf_cfg, meta_cfg, seed_L, rounds, batch_size,
                              aug_K, alpha, per_class_min, refresh_every, al_method, logger,
                              out_dir, log_jsonl=str(Path(out_dir)/"sim_log.jsonl") if out_dir else None)
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

    seed_cfg = cfg.get("seeding", {})
    seed_method = seed_cfg.get("method", "k_coreset")
    seed_L = {}
    for cid, pool in pools.items():
        X = pool["emb"][pool["originals"]]
        if seed_method == "kmeans":
            idx_rel = kmeans_seed(X, k=min(seed_k, X.shape[0]), random_state=rs)
        else:
            idx_rel = kcenter_greedy(X, k=min(seed_k, X.shape[0]), seed=rs)
        seed_L[cid] = [int(pool["originals"][i]) for i in idx_rel]

    out_dir = cfg.get("logging", {}).get("out_dir")

    run_sim(client_pools=pools, classes=classes, rf_cfg=rf_cfg, meta_cfg=meta_cfg,
            seed_L=seed_L, rounds=rounds, batch_size=batch_size, aug_K=aug_K, alpha=0.0,
            per_class_min=per_class_min, refresh_every=int(meta_cfg.get("refresh_every",1)),
            al_method=al_method, out_dir=out_dir)
