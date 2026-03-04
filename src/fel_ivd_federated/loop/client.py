import numpy as np
from sklearn.preprocessing import StandardScaler
from ..models.rf import RFWrapper
from ..meta.summary_features import summary_features_from_tree_probs, entropy
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

class ClientState:
    def __init__(self, client_id, classes, emb, y, axial, originals, augs_by_axial, idx_by_axial,
                 rf_cfg, al_method="qbc", train_aug_K=0, tta_n=0, meta=None):
        self.client_id = client_id
        self.classes = classes
        self.emb = emb
        self.y_pool = np.asarray(y)
        self.axial = np.asarray(axial)
        self.originals = sorted(list(originals))
        self.augs_by_axial = augs_by_axial
        self.idx_by_axial = idx_by_axial
        self.L = []
        self.U = sorted(self.originals.copy())
        self.scaler = StandardScaler()
        self.rf = RFWrapper(**rf_cfg)
        self.al_method = al_method
        self.train_aug_K = int(train_aug_K)
        self.tta_n = int(tta_n)
        self.meta = meta

    def fit_rf_with_aug(self, K):
        if len(self.L) == 0:
            return
        Z_L = self.emb[self.L]
        self.scaler.fit(Z_L)

        X_list = [Z_L]
        y_list = [self.y_pool[self.L]]
        for idx in self.L:
            ax = self.axial[idx]
            augs = self.augs_by_axial.get(ax, [])[:min(K, len(self.augs_by_axial.get(ax, [])))]
            if augs:
                X_list.append(self.emb[augs])
                y_list.append(self.y_pool[augs])
        Xtr = np.vstack(X_list)
        ytr = np.concatenate(y_list)
        Ztr = self.scaler.transform(Xtr)
        self.rf.fit(Ztr, ytr)

    def _rf_proba(self, Z):
        return self.rf.predict_proba(Z)

    def _meta_proba(self, Z):
        if self.meta is None:
            return self._rf_proba(Z)
        trees = self.rf._clf.estimators_
        tree_probs = [t.predict_proba(Z) for t in trees]
        Xf, _ = summary_features_from_tree_probs(tree_probs, self.classes)
        return self.meta.predict_proba(Xf)

    def _committee_mean(self, Z):
        trees = self.rf._clf.estimators_
        P = np.stack([t.predict_proba(Z) for t in trees], axis=0)
        return P.mean(axis=0), P

    def eval_split(self, split="U", use_meta=False):
        if split == "U":
            idx = self.U
        elif split == "L":
            idx = self.L
        else:
            idx = list(range(len(self.y_pool)))
        if not idx:
            return {"balanced_acc": None, "macro_f1": None, "ovr_auc": None, "n": 0}

        Z = self.scaler.transform(self.emb[idx])
        P = self._meta_proba(Z) if use_meta else self._rf_proba(Z)
        y_true = self.y_pool[idx]
        y_pred = np.array([self.classes[i] for i in P.argmax(axis=1)])

        out = {"n": len(idx)}
        try: out["balanced_acc"] = float(balanced_accuracy_score(y_true, y_pred))
        except: out["balanced_acc"] = None
        try: out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
        except: out["macro_f1"] = None
        try: out["ovr_auc"] = float(roc_auc_score(y_true, P, multi_class="ovr", labels=self.classes))
        except: out["ovr_auc"] = None
        return out

    def acquire(self, B, alpha=0.0, per_class_min=1, method="qbc"):
        if B <= 0 or len(self.U) == 0:
            return {"picked_idx": [], "picked_counts": {}, "train_rows_aug": int(len(self.L)),
                    "committee_size": len(self.rf._clf.estimators_) if self.rf._clf is not None else 0}
        ZU = self.scaler.transform(self.emb[self.U])
        mean_p, tree_P = self._committee_mean(ZU)
        
        if method == "entropy":
            scores = entropy(mean_p)
        elif method == "margin":
            part = np.partition(-mean_p, 2, axis=1)
            top1 = -part[:, 0]; top2 = -part[:, 1]
            scores = 1.0 - (top1 - top2)
        elif method == "least_confident":
            scores = 1.0 - mean_p.max(axis=1)
        elif method == "bald":
            scores = entropy(mean_p) - entropy(tree_P).mean(axis=0)
        else:
            eps = 1e-12
            m = np.clip(mean_p[None, :, :], eps, 1.0)
            P = np.clip(tree_P, eps, 1.0)
            M = 0.5*(P + m)
            js = 0.5*((P*np.log(P/M)).sum(axis=2) + (m*np.log(m/M)).sum(axis=2))
            scores = js.mean(axis=0)

        preds = mean_p.argmax(axis=1)
        cls_list = self.classes
        taken = set()
        order = np.argsort(-scores, kind="mergesort")
        per_class_taken = {c:0 for c in cls_list}
        for i in order:
            c = cls_list[preds[i]]
            if per_class_taken[c] < per_class_min:
                taken.add(i); per_class_taken[c] += 1
            if len(taken) >= min(B, len(self.U)):
                break
        for i in order:
            if len(taken) >= min(B, len(self.U)): break
            if i not in taken:
                taken.add(i)

        picked_rel = sorted(list(taken))
        picked_idx = [self.U[i] for i in picked_rel]

        picked_counts = {}
        for idx in picked_idx:
            cls = self.y_pool[idx]
            picked_counts[cls] = picked_counts.get(cls, 0) + 1

        self.L += picked_idx
        self.L = sorted(self.L)
        self.U = sorted([i for i in self.U if i not in set(self.L)])

        n_aug = 0
        for i in self.L:
            ax = self.axial[i]
            n_aug += min(self.train_aug_K, len(self.augs_by_axial.get(ax, [])))
        train_rows_aug = len(self.L) + n_aug

        return {
            "picked_idx": picked_idx,
            "picked_counts": picked_counts,
            "train_rows_aug": int(train_rows_aug),
            "committee_size": len(self.rf._clf.estimators_)
        }
