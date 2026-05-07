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

def _pad_tree_proba(tree, Z, all_classes, forest_classes=None):
    """Get predict_proba from a single tree, padded to all_classes columns.
    
    sklearn RF trees use integer-encoded labels internally.
    If forest_classes is provided (the forest's .classes_ attribute),
    we use it to map tree integer classes back to original labels.
    """
    P = tree.predict_proba(Z)
    tree_cls = tree.classes_
    n_all = len(all_classes)
    if len(tree_cls) == n_all:
        return P
    full = np.zeros((P.shape[0], n_all), dtype=P.dtype)
    for i, tc in enumerate(tree_cls):
        # tc is an integer index into forest_classes
        if forest_classes is not None:
            label = forest_classes[int(tc)]
            if label in all_classes:
                j = all_classes.index(label)
                full[:, j] = P[:, i]
        else:
            j = int(tc)
            if 0 <= j < n_all:
                full[:, j] = P[:, i]
    return full

class FedRidgeMeta:
    def __init__(self, classes, lam=1e-2, oof_folds=5, include_T=False,
                 aug_mode='off', tta_n=0, feature_groups=None):
        self.classes = classes
        self.lam = float(lam)
        self.oof_folds = int(oof_folds)
        self.include_T = bool(include_T)
        self.aug_mode = aug_mode
        self.tta_n = int(tta_n)
        self.feature_groups = feature_groups  # None → all
        self.W = None
        self.mu = None
        self.sigma = None
        self.with_bias = True

    @staticmethod
    def _zscore(X):
        mu = X.mean(axis=0, keepdims=True)
        sigma = X.std(axis=0, keepdims=True) + 1e-6
        Xn = (X - mu) / sigma
        return Xn, mu, sigma

    def client_oof_stats(self, X_base, y_str, tree_list, scaler,
                         group_ids=None, aug_mode='off', forest_classes=None):
        """OOF stats using per-tree predictions (RF/GBT only)."""
        y = np.asarray(y_str)
        counts = Counter(y)
        min_cls = min(counts.values()) if counts else 0
        n_splits = int(min(self.oof_folds, max(min_cls, 1)))
        if n_splits < 2:
            return None

        if aug_mode == 'grouped' and group_ids is not None:
            skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
            split_iter = skf.split(X_base, y, groups=group_ids)
        else:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            split_iter = skf.split(X_base, y)

        feats, labels = [], []
        for tr, te in split_iter:
            Zte = scaler.transform(X_base[te])
            tree_probs = [_pad_tree_proba(t, Zte, self.classes, forest_classes=forest_classes) for t in tree_list]
            Xf, _ = summary_features_from_tree_probs(tree_probs, self.classes,
                                                        feature_groups=self.feature_groups)
            feats.append(Xf); labels.append(y[te])

        X_feat = np.vstack(feats)
        Y = one_hot(np.concatenate(labels), self.classes)

        Xn, mu_local, sigma_local = self._zscore(X_feat)
        ones = np.ones((Xn.shape[0], 1), dtype=Xn.dtype)
        Xnb = np.concatenate([Xn, ones], axis=1)

        Sxx = Xnb.T @ Xnb
        Sxy = Xnb.T @ Y

        var_local = (sigma_local ** 2)
        n_rows = X_feat.shape[0]

        return (Sxx, Sxy, mu_local.astype(np.float32), var_local.astype(np.float32), int(n_rows))

    def client_oof_stats_generic(self, X_base, y_str, model, scaler):
        """OOF stats for non-tree models.

        Unlike tree models where we can cheaply iterate individual estimators
        on held-out folds, non-tree models (SVM, MLP, GBT, LR) would require
        cloning and re-fitting on each fold — prohibitively expensive.

        Instead, we use the already-fitted model's predictions directly on the
        training data, matching the RF path where fitted trees predict on folds
        they partially trained on. The ridge regularisation (λ) prevents
        overfitting the meta-learner to these in-sample predictions.
        """
        from .summary_features import summary_features_from_proba

        y = np.asarray(y_str)
        if len(y) < 2:
            return None

        Z = scaler.transform(X_base)
        P = model.predict_proba(Z)

        # Pad to full class set if needed
        seen = list(model._clf.classes_)
        if len(seen) < len(self.classes):
            full = np.zeros((P.shape[0], len(self.classes)), dtype=P.dtype)
            for i, c in enumerate(seen):
                if c in self.classes:
                    j = self.classes.index(c)
                    full[:, j] = P[:, i]
            P = full

        Xf, _ = summary_features_from_proba(P, self.classes)
        Y = one_hot(y, self.classes)

        Xn, mu_local, sigma_local = self._zscore(Xf)
        ones = np.ones((Xn.shape[0], 1), dtype=Xn.dtype)
        Xnb = np.concatenate([Xn, ones], axis=1)

        Sxx = Xnb.T @ Xnb
        Sxy = Xnb.T @ Y

        var_local = (sigma_local ** 2)
        n_rows = Xf.shape[0]

        return (Sxx, Sxy, mu_local.astype(np.float32), var_local.astype(np.float32), int(n_rows))

    def server_solve(self, Sxx, Sxy, mu_global, sigma_global, with_bias=True):
        self.mu = mu_global.astype(np.float32)
        self.sigma = sigma_global.astype(np.float32) + 1e-6
        self.with_bias = bool(with_bias)
        Dp1 = Sxx.shape[0]
        A = Sxx + self.lam * np.eye(Dp1, dtype=np.float32)
        self.W = np.linalg.solve(A, Sxy)

    def predict_proba(self, X_feat):
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
