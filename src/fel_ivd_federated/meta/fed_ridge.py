import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from collections import Counter
from .summary_features import summary_features_from_tree_probs

def one_hot(y, classes):
    idx = {c: i for i, c in enumerate(classes)}
    M = np.zeros((len(y), len(classes)), dtype=np.float32)
    for i, v in enumerate(y):
        M[i, idx[v]] = 1.0
    return M

class FedRidgeMeta:
    def __init__(self, classes, lam=1e-2, oof_folds=5, include_T=False,
                 aug_mode='off', tta_n=0):
        self.classes = classes
        self.lam = float(lam)
        self.oof_folds = int(oof_folds)
        self.include_T = bool(include_T)
        self.aug_mode = aug_mode      # 'off' | 'tta' | 'grouped'
        self.tta_n = int(tta_n)

        # learned parameters
        self.W = None                 # [D(+1), C]
        self.mu = None                # [1, D]
        self.sigma = None             # [1, D]
        self.with_bias = True         # we will append a constant 1 feature

    # ---------- TRAIN SIDE (client) ----------

    @staticmethod
    def _zscore(X):
        mu = X.mean(axis=0, keepdims=True)
        sigma = X.std(axis=0, keepdims=True) + 1e-6
        Xn = (X - mu) / sigma
        return Xn, mu, sigma

    def client_oof_stats(self, X_base, y_str, tree_list, scaler,
                         group_ids=None, aug_mode='off'):
        """
        Build OOF features on *labeled originals* and return sufficient stats.

        Returns:
            Sxx      : [D(+1), D(+1)]  (z-scored, with bias column)
            Sxy      : [D(+1), C]
            mu_local : [1, D]          (pre-zscore mean on *this client*)
            var_local: [1, D]          (pre-zscore variance on *this client*)
            n_rows   : int             (total OOF rows on this client)
        """
        y = np.asarray(y_str)
        counts = Counter(y)
        min_cls = min(counts.values()) if counts else 0
        n_splits = int(min(self.oof_folds, max(min_cls, 1)))
        if n_splits < 2:
            return None  # not enough labels to do OOF

        if aug_mode == 'grouped' and group_ids is not None:
            skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
            split_iter = skf.split(X_base, y, groups=group_ids)
        else:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            split_iter = skf.split(X_base, y)

        feats, labels = [], []
        for tr, te in split_iter:
            Zte = scaler.transform(X_base[te])         # scale using client's RF scaler
            tree_probs = [t.predict_proba(Zte) for t in tree_list]
            Xf, _ = summary_features_from_tree_probs(tree_probs, self.classes)
            feats.append(Xf); labels.append(y[te])

        X_feat = np.vstack(feats)                      # [N, D]
        Y = one_hot(np.concatenate(labels), self.classes)  # [N, C]

        # ---- standardize on this client's concatenated OOF features
        Xn, mu_local, sigma_local = self._zscore(X_feat)

        # ---- append bias column (constant 1)
        ones = np.ones((Xn.shape[0], 1), dtype=Xn.dtype)
        Xnb = np.concatenate([Xn, ones], axis=1)       # [N, D+1]

        # ---- sufficient stats on standardized + bias features
        Sxx = Xnb.T @ Xnb                               # [D+1, D+1]
        Sxy = Xnb.T @ Y                                 # [D+1, C]

        # return local moments so the server can pool a global mu/sigma
        var_local = (sigma_local ** 2)
        n_rows = X_feat.shape[0]

        return (Sxx, Sxy, mu_local.astype(np.float32), var_local.astype(np.float32), int(n_rows))

    # ---------- TRAIN SIDE (server) ----------

    def server_solve(self, Sxx, Sxy, mu_global, sigma_global, with_bias=True):
        """
        Finalize META:
          - set global μ/σ for inference-time standardization
          - solve ridge on pre-accumulated Sxx/Sxy (already z-scored on clients)
        """
        self.mu = mu_global.astype(np.float32)          # [1, D]
        self.sigma = sigma_global.astype(np.float32) + 1e-6
        self.with_bias = bool(with_bias)

        Dp1 = Sxx.shape[0]                              # D (+1 if bias)
        A = Sxx + self.lam * np.eye(Dp1, dtype=np.float32)
        self.W = np.linalg.solve(A, Sxy)               # [D+1, C]

    # ---------- INFERENCE ----------

    def predict_proba(self, X_feat):
        """
        Apply the SAME transform used in training:
          - z-score with stored μ/σ
          - append bias if used
          - linear -> softmax
        """
        C = len(self.classes)
        n = X_feat.shape[0]
        if self.W is None or self.mu is None or self.sigma is None:
            return np.full((n, C), 1.0 / C, dtype=np.float32)

        Xn = (X_feat - self.mu) / self.sigma
        if self.with_bias:
            Xn = np.concatenate([Xn, np.ones((n, 1), dtype=Xn.dtype)], axis=1)

        logits = Xn @ self.W
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
