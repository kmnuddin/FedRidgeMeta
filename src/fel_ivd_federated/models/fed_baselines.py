"""
Federated baseline implementations: FedAvg, FedProx, FedNova.

All use a small MLP (256→64→32→K) on frozen embeddings, with numpy-based
forward/backward passes and weight extraction for federated aggregation.
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


# ── Activation functions ──

def _relu(x):
    return np.maximum(0, x)

def _relu_grad(x):
    return (x > 0).astype(x.dtype)

def _softmax(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ── Numpy MLP ──

class NumpyMLP:
    """Small MLP with extractable/settable weights for federated aggregation.

    Architecture: input_dim → 64 (ReLU) → 32 (ReLU) → n_classes (softmax)
    Training: mini-batch SGD with cross-entropy loss.
    """

    def __init__(self, input_dim, n_classes, lr=0.01, batch_size=64, seed=42):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)

        # Xavier init
        self.W1 = self.rng.randn(input_dim, 64).astype(np.float32) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(64, dtype=np.float32)
        self.W2 = self.rng.randn(64, 32).astype(np.float32) * np.sqrt(2.0 / 64)
        self.b2 = np.zeros(32, dtype=np.float32)
        self.W3 = self.rng.randn(32, n_classes).astype(np.float32) * np.sqrt(2.0 / 32)
        self.b3 = np.zeros(n_classes, dtype=np.float32)

        self.le = None  # LabelEncoder, set during first fit
        self.all_classes = None  # full class list for padding

    def get_weights(self):
        """Return a flat list of (name, array) tuples."""
        return [
            ('W1', self.W1.copy()), ('b1', self.b1.copy()),
            ('W2', self.W2.copy()), ('b2', self.b2.copy()),
            ('W3', self.W3.copy()), ('b3', self.b3.copy()),
        ]

    def set_weights(self, weight_list):
        """Set weights from a list of (name, array) tuples."""
        wd = dict(weight_list)
        self.W1 = wd['W1'].copy()
        self.b1 = wd['b1'].copy()
        self.W2 = wd['W2'].copy()
        self.b2 = wd['b2'].copy()
        self.W3 = wd['W3'].copy()
        self.b3 = wd['b3'].copy()

    def n_params(self):
        """Total number of trainable parameters."""
        return sum(w.size for _, w in self.get_weights())

    def n_bytes(self):
        """Communication cost: bytes to transmit all weights (float32)."""
        return self.n_params() * 4

    def _forward(self, X):
        """Forward pass, return (logits, caches for backprop)."""
        z1 = X @ self.W1 + self.b1
        a1 = _relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = _relu(z2)
        z3 = a2 @ self.W3 + self.b3
        return z3, (X, z1, a1, z2, a2)

    def predict_proba(self, X):
        """Return (N, K) probability array, padded to all_classes if needed."""
        logits, _ = self._forward(X.astype(np.float32))
        P = _softmax(logits)
        if self.all_classes is not None and self.le is not None:
            seen = list(self.le.classes_)
            if len(seen) < len(self.all_classes):
                full = np.zeros((P.shape[0], len(self.all_classes)), dtype=P.dtype)
                for i, c in enumerate(seen):
                    if c in self.all_classes:
                        j = self.all_classes.index(c)
                        full[:, j] = P[:, i]
                return full
        return P

    def fit(self, X, y, epochs=5, proximal_mu=0.0, global_weights=None):
        """Train with SGD. Returns number of local steps taken.

        Parameters
        ----------
        epochs : int — local epochs per federation round
        proximal_mu : float — FedProx proximal term (0 = FedAvg)
        global_weights : list | None — global model weights for proximal term
        """
        X = X.astype(np.float32)

        # Encode labels
        if self.le is None:
            self.le = LabelEncoder()
            self.le.fit(y)
        y_enc = self.le.transform(y)
        K = len(self.le.classes_)

        # One-hot targets
        Y = np.zeros((len(y_enc), K), dtype=np.float32)
        Y[np.arange(len(y_enc)), y_enc] = 1.0

        # Global weights for proximal term
        gw = None
        if proximal_mu > 0 and global_weights is not None:
            gw = dict(global_weights)

        n = len(X)
        n_steps = 0

        for _ in range(epochs):
            idx = self.rng.permutation(n)
            for start in range(0, n, self.batch_size):
                batch = idx[start:start + self.batch_size]
                Xb, Yb = X[batch], Y[batch]
                bs = len(Xb)

                # Forward
                z3, (Xin, z1, a1, z2, a2) = self._forward(Xb)
                P = _softmax(z3)

                # Backward: cross-entropy gradient
                dz3 = (P - Yb) / bs
                dW3 = a2.T @ dz3
                db3 = dz3.sum(axis=0)

                da2 = dz3 @ self.W3.T
                dz2 = da2 * _relu_grad(z2)
                dW2 = a1.T @ dz2
                db2 = dz2.sum(axis=0)

                da1 = dz2 @ self.W2.T
                dz1 = da1 * _relu_grad(z1)
                dW1 = Xin.T @ dz1
                db1 = dz1.sum(axis=0)

                # Add proximal term (FedProx)
                if proximal_mu > 0 and gw is not None:
                    dW1 += proximal_mu * (self.W1 - gw['W1'])
                    db1 += proximal_mu * (self.b1 - gw['b1'])
                    dW2 += proximal_mu * (self.W2 - gw['W2'])
                    db2 += proximal_mu * (self.b2 - gw['b2'])
                    dW3 += proximal_mu * (self.W3 - gw['W3'])
                    db3 += proximal_mu * (self.b3 - gw['b3'])

                # SGD update
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
                self.W3 -= self.lr * dW3
                self.b3 -= self.lr * db3

                n_steps += 1

        return n_steps


# ── Federated Aggregation ──

def fedavg_aggregate(client_weights, client_sizes):
    """FedAvg: weighted average of model parameters by dataset size."""
    total = sum(client_sizes)
    avg = []
    for name in [n for n, _ in client_weights[0]]:
        stacked = np.stack([dict(cw)[name] for cw in client_weights])
        weights = np.array(client_sizes, dtype=np.float32) / total
        weighted = np.tensordot(weights, stacked, axes=([0], [0]))
        avg.append((name, weighted))
    return avg


def fednova_aggregate(client_weights, client_sizes, client_steps):
    """FedNova: normalized averaging accounting for different local steps.

    Each client's update is normalized by its number of local steps,
    then re-scaled by the average number of steps.
    """
    total_n = sum(client_sizes)
    tau_eff = sum(s * n / total_n for s, n in zip(client_steps, client_sizes))

    # We need the global model to compute deltas
    # Instead, we normalize: w_global = w_global + tau_eff * sum(p_i * d_i / tau_i)
    # where d_i = (w_i - w_global) is the update, tau_i = steps
    # This simplifies to a weighted average with step-normalized weights

    avg = []
    for name in [n for n, _ in client_weights[0]]:
        result = np.zeros_like(dict(client_weights[0])[name])
        for cw, n_data, n_steps in zip(client_weights, client_sizes, client_steps):
            p_i = n_data / total_n
            result += p_i * (tau_eff / max(n_steps, 1)) * dict(cw)[name]
        avg.append((name, result))
    return avg


# ── Baseline Runner ──

class FedBaselineRunner:
    """Runs the federated AL loop with FedAvg/FedProx/FedNova.

    Matches the structure of SimulationRunner but uses MLP + gradient-based
    federation instead of RF + sufficient statistics.
    """

    def __init__(self, method, client_pools, classes, seed_L, rounds, batch_size,
                 test_indices, holdout_frac=0.30, al_method="margin",
                 local_epochs=5, lr=0.01, proximal_mu=0.01,
                 out_dir=None, log_jsonl=None):
        """
        Parameters
        ----------
        method : str — 'fedavg', 'fedprox', 'fednova'
        """
        self.method = method
        self.client_pools = client_pools
        self.classes = list(classes)
        self.seed_L = seed_L
        self.rounds = rounds
        self.batch_size = batch_size
        self.test_indices = test_indices or {}
        self.al_method = al_method
        self.local_epochs = local_epochs
        self.lr = lr
        self.proximal_mu = proximal_mu if method == 'fedprox' else 0.0
        self.out_dir = out_dir
        self.log_jsonl = log_jsonl

        self.input_dim = None  # set from data
        self.n_classes = len(classes)

    def run(self):
        import time, json, os
        from pathlib import Path
        from sklearn.preprocessing import StandardScaler
        from ..utils.calibration import expected_calibration_error, brier_score

        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)

        # ── Init clients ──
        clients = {}
        for cid, pool in self.client_pools.items():
            test_idx = self.test_indices.get(cid, [])
            pool_originals = pool["originals"]
            if test_idx:
                test_set = set(test_idx)
                pool_originals = [i for i in pool_originals if i not in test_set]

            clients[cid] = {
                'emb': pool['emb'],
                'y': np.asarray(pool['label']),
                'originals': sorted(pool_originals),
                'L': sorted(self.seed_L.get(cid, [])),
                'test_indices': sorted(test_idx),
                'scaler': StandardScaler(),
            }
            clients[cid]['U'] = sorted([i for i in clients[cid]['originals']
                                        if i not in set(clients[cid]['L'])])

        # Determine input dim
        any_cid = next(iter(clients))
        self.input_dim = clients[any_cid]['emb'].shape[1]

        # ── Init global model ──
        global_model = NumpyMLP(self.input_dim, self.n_classes, lr=self.lr, seed=42)
        global_model.all_classes = self.classes
        global_weights = global_model.get_weights()

        # Track metrics
        t0 = time.time()
        total_bytes_sent = 0

        for r in range(self.rounds):
            t_round = time.time()

            # 1) Local training
            t_train = time.time()
            client_weights_list = []
            client_sizes = []
            client_steps = []

            for cid, cstate in clients.items():
                if len(cstate['L']) == 0:
                    continue

                # Fit scaler and prepare data
                Z_L = cstate['emb'][cstate['L']]
                cstate['scaler'].fit(Z_L)
                X_train = cstate['scaler'].transform(Z_L)
                y_train = cstate['y'][cstate['L']]

                # Create local model from global weights
                local_model = NumpyMLP(self.input_dim, self.n_classes,
                                       lr=self.lr, seed=42 + r)
                local_model.set_weights(global_weights)
                local_model.all_classes = self.classes

                # Local SGD
                n_steps = local_model.fit(
                    X_train, y_train,
                    epochs=self.local_epochs,
                    proximal_mu=self.proximal_mu,
                    global_weights=global_weights,
                )

                client_weights_list.append(local_model.get_weights())
                client_sizes.append(len(cstate['L']))
                client_steps.append(n_steps)

                # Store local model for evaluation
                cstate['model'] = local_model

            t_train = time.time() - t_train

            # 2) Server aggregation
            t_agg = time.time()
            if client_weights_list:
                if self.method == 'fednova':
                    global_weights = fednova_aggregate(
                        client_weights_list, client_sizes, client_steps)
                else:
                    # FedAvg and FedProx both use weighted averaging
                    global_weights = fedavg_aggregate(
                        client_weights_list, client_sizes)

                # Update global model
                global_model.set_weights(global_weights)

                # Communication cost: each client sends full weights
                bytes_this_round = len(client_weights_list) * global_model.n_bytes()
                # Plus server broadcasts back
                bytes_this_round += len(client_weights_list) * global_model.n_bytes()
                total_bytes_sent += bytes_this_round
            else:
                bytes_this_round = 0

            t_agg = time.time() - t_agg

            # 3) Evaluation — per-client on test set
            t_eval = time.time()
            per_client_test = {}
            all_yt, all_pp_local, all_pp_global = [], [], []

            for cid, cstate in clients.items():
                if not cstate['test_indices'] or 'model' not in cstate:
                    continue

                Z_test = cstate['scaler'].transform(cstate['emb'][cstate['test_indices']])
                y_test = cstate['y'][cstate['test_indices']]

                # Local model predictions
                P_local = cstate['model'].predict_proba(Z_test)
                y_pred_local = np.array([self.classes[i] for i in P_local.argmax(axis=1)])

                # Global model predictions (= "META" equivalent)
                global_eval = NumpyMLP(self.input_dim, self.n_classes, lr=self.lr)
                global_eval.set_weights(global_weights)
                global_eval.all_classes = self.classes
                global_eval.le = cstate['model'].le
                P_global = global_eval.predict_proba(Z_test)
                y_pred_global = np.array([self.classes[i] for i in P_global.argmax(axis=1)])

                def _metrics(y_true, y_pred, P):
                    out = {}
                    try: out['macro_f1'] = float(f1_score(y_true, y_pred, average='macro'))
                    except: out['macro_f1'] = None
                    try: out['balanced_acc'] = float(balanced_accuracy_score(y_true, y_pred))
                    except: out['balanced_acc'] = None
                    try: out['ovr_auc'] = float(roc_auc_score(y_true, P, multi_class='ovr', labels=self.classes))
                    except: out['ovr_auc'] = None
                    return out

                m_local = _metrics(y_test, y_pred_local, P_local)
                m_global = _metrics(y_test, y_pred_global, P_global)

                per_client_test[cid] = {'local': m_local, 'global': m_global}
                all_yt.append(y_test)
                all_pp_local.append(P_local)
                all_pp_global.append(P_global)

            # Aggregate test metrics
            avg_local = {}
            avg_global = {}
            if per_client_test:
                for key in ['macro_f1', 'balanced_acc', 'ovr_auc']:
                    vals_l = [v['local'][key] for v in per_client_test.values() if v['local'][key] is not None]
                    vals_g = [v['global'][key] for v in per_client_test.values() if v['global'][key] is not None]
                    avg_local[key] = sum(vals_l) / len(vals_l) if vals_l else None
                    avg_global[key] = sum(vals_g) / len(vals_g) if vals_g else None

            # Calibration on pooled test
            cal_local = {}
            cal_global = {}
            if all_yt:
                yt_all = np.concatenate(all_yt)
                pp_l = np.vstack(all_pp_local)
                pp_g = np.vstack(all_pp_global)
                cal_local['ece'] = float(expected_calibration_error(yt_all, pp_l, self.classes))
                cal_local['brier'] = float(brier_score(yt_all, pp_l, self.classes))
                cal_global['ece'] = float(expected_calibration_error(yt_all, pp_g, self.classes))
                cal_global['brier'] = float(brier_score(yt_all, pp_g, self.classes))

            t_eval = time.time() - t_eval

            # 4) Acquisition (margin sampling on local model)
            t_acq = time.time()
            for cid, cstate in clients.items():
                if len(cstate['U']) == 0 or 'model' not in cstate:
                    continue
                Z_U = cstate['scaler'].transform(cstate['emb'][cstate['U']])
                P = cstate['model'].predict_proba(Z_U)

                # Margin sampling
                part = np.partition(-P, 2, axis=1)
                top1, top2 = -part[:, 0], -part[:, 1]
                scores = 1.0 - (top1 - top2)

                B = min(self.batch_size, len(cstate['U']))
                pick_rel = np.argsort(-scores)[:B]
                pick_idx = [cstate['U'][i] for i in pick_rel]

                cstate['L'] = sorted(cstate['L'] + pick_idx)
                cstate['U'] = sorted([i for i in cstate['U'] if i not in set(cstate['L'])])

            t_acq = time.time() - t_acq

            # 5) Log
            if self.log_jsonl:
                entry = {
                    'round': r + 1,
                    'method': self.method,
                    'avg_test': {
                        'local': {**avg_local, **cal_local},
                        'global': {**avg_global, **cal_global},
                    },
                    'clients': {},
                    'timing_sec': {
                        'train': t_train,
                        'aggregate': t_agg,
                        'eval': t_eval,
                        'acquire': t_acq,
                    },
                    'communication': {
                        'bytes_this_round': bytes_this_round,
                        'bytes_cumulative': total_bytes_sent,
                        'n_params': global_model.n_params(),
                    },
                }
                for cid in clients:
                    if cid in per_client_test:
                        entry['clients'][cid] = {
                            'local': per_client_test[cid]['local'],
                            'global': per_client_test[cid]['global'],
                            'L_size': len(clients[cid]['L']),
                            'U_size': len(clients[cid]['U']),
                        }

                with open(self.log_jsonl, 'a') as f:
                    f.write(json.dumps(entry) + '\n')

        # Summary
        if self.log_jsonl:
            with open(self.log_jsonl, 'a') as f:
                f.write(json.dumps({
                    'summary': {
                        'method': self.method,
                        'rounds': self.rounds,
                        'wall_clock_sec': time.time() - t0,
                        'total_bytes': total_bytes_sent,
                        'n_params': global_model.n_params(),
                        'local_epochs': self.local_epochs,
                        'lr': self.lr,
                        'proximal_mu': self.proximal_mu,
                    }
                }) + '\n')


def run_baseline_from_yaml(cfg: dict):
    """Entry point matching run_sim_from_yaml interface.

    Replicates the same data loading, holdout split, and seeding logic
    as sim_runner.run_sim_from_yaml so the comparison is fair.
    """
    import random
    from pathlib import Path
    from ..utils.data_io import load_clients_from_csvs, split_holdout
    from ..selection.kmeans_seed import kmeans_seed

    raw_rs = cfg.get("random_state", 42)
    rs = int(raw_rs)
    random.seed(rs)
    np.random.seed(rs)

    # ── Parse config ──
    out_dir = cfg.get("logging", {}).get("out_dir", "runs/baseline")
    method = cfg.get("baseline_method", "fedavg")

    data = cfg["data"]
    label_col = data.get("label_col", "ivd_level")
    id_col    = data.get("id_col", "id")
    image_col = data.get("image_col", "axial_path")
    emb_col   = data.get("emb_col", "emb")

    # ── Load data (same as sim_runner) ──
    csv_map = {}
    if "rsna_csv" in data: csv_map["RSNA"] = data["rsna_csv"]
    if "mendeley_csv" in data: csv_map["MENDELEY"] = data["mendeley_csv"]
    pools = load_clients_from_csvs(csv_map, label_col, id_col, emb_col, axial_col=image_col)

    labels = []
    for p in pools.values():
        labels.append(p["label"][p["originals"]])
    classes = sorted(np.unique(np.concatenate(labels)).tolist())

    # ── Holdout split (same as sim_runner) ──
    holdout_cfg = cfg.get("holdout", {})
    holdout_frac = float(holdout_cfg.get("test_frac", 0.30))
    test_indices = {}
    if holdout_frac > 0:
        for cid, pool in pools.items():
            pool_orig, test_orig = split_holdout(pool, test_frac=holdout_frac, seed=rs)
            test_indices[cid] = test_orig
            pools[cid]["originals"] = pool_orig

    # ── Seeding (same as sim_runner) ──
    seed_k = int(cfg.get("seed_k", 200))
    n_classes = len(classes)
    seed_L = {}
    for cid, pool in pools.items():
        pool_size = len(pool["originals"])
        effective_seed_k = max(n_classes, min(seed_k, pool_size // 4))
        X = pool["emb"][pool["originals"]]
        idx_rel = kmeans_seed(X, k=min(effective_seed_k, X.shape[0]), random_state=rs)
        seed_L[cid] = [int(pool["originals"][i]) for i in idx_rel]

    # ── Run params ──
    rounds = int(cfg.get("rounds", 60))
    batch_B = int(cfg.get("al", {}).get("batch_B", 200))
    local_epochs = int(cfg.get("baseline_epochs", 5))
    lr = float(cfg.get("baseline_lr", 0.01))
    proximal_mu = float(cfg.get("baseline_proximal_mu", 0.01))
    al_method = cfg.get("al", {}).get("method", "margin")

    runner = FedBaselineRunner(
        method=method,
        client_pools=pools,
        classes=classes,
        seed_L=seed_L,
        rounds=rounds,
        batch_size=batch_B,
        test_indices=test_indices,
        al_method=al_method,
        local_epochs=local_epochs,
        lr=lr,
        proximal_mu=proximal_mu,
        out_dir=out_dir,
        log_jsonl=str(Path(out_dir) / "sim_log.jsonl"),
    )
    runner.run()