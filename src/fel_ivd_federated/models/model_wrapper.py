"""Generic model wrappers with a common interface: fit(X, y), predict_proba(X).

All wrappers produce (N, K) probability arrays padded to `all_classes`.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


class _BaseWrapper:
    """Common padding logic shared by all model types."""

    def __init__(self):
        self._clf = None
        self.all_classes = None  # set externally to the full class list

    def _pad_proba(self, P):
        if self.all_classes is None:
            return P
        seen = list(self._clf.classes_)
        if len(seen) == len(self.all_classes):
            return P
        full = np.zeros((P.shape[0], len(self.all_classes)), dtype=P.dtype)
        for i, c in enumerate(seen):
            if c in self.all_classes:
                j = self.all_classes.index(c)
                full[:, j] = P[:, i]
        return full

    def fit(self, Z, y):
        self._clf.fit(Z, y)

    def predict_proba(self, Z):
        P = self._clf.predict_proba(Z)
        return self._pad_proba(P)

    @property
    def has_estimators(self):
        """True if model has iteratable base learners (RF, GBT)."""
        return False


class RFModel(_BaseWrapper):
    def __init__(self, n_estimators=100, max_depth=None, max_features='sqrt',
                 class_weight='balanced_subsample', n_jobs=-1, **kwargs):
        super().__init__()
        self._clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            max_features=max_features, class_weight=class_weight,
            n_jobs=n_jobs, random_state=42,
        )

    @property
    def has_estimators(self):
        return True


class GBTModel(_BaseWrapper):
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1,
                 subsample=0.8, **kwargs):
        super().__init__()
        self._clf = GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample,
            random_state=42,
        )

    @property
    def has_estimators(self):
        # GBT's estimators_ are DecisionTreeRegressors, not classifiers.
        # Cannot iterate for per-tree class probabilities. Use generic OOF.
        return False


class LogisticModel(_BaseWrapper):
    def __init__(self, C=1.0, max_iter=1000, **kwargs):
        super().__init__()
        self._clf = LogisticRegression(
            C=C, max_iter=max_iter,
            solver='lbfgs', random_state=42,
        )


class SVMModel(_BaseWrapper):
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', **kwargs):
        super().__init__()
        self._clf = SVC(
            C=C, kernel=kernel, gamma=gamma,
            probability=True,  # enables Platt scaling
            random_state=42,
        )


class MLPModel(_BaseWrapper):
    def __init__(self, hidden_layers=(64, 32), max_iter=500,
                 learning_rate_init=0.001, **kwargs):
        super().__init__()
        self._clf = MLPClassifier(
            hidden_layer_sizes=hidden_layers, max_iter=max_iter,
            learning_rate_init=learning_rate_init,
            early_stopping=False,
            random_state=42,
        )


# Registry for sweep configs
MODEL_REGISTRY = {
    'rf':       RFModel,
    'gbt':      GBTModel,
    'logistic': LogisticModel,
    'svm':      SVMModel,
    'mlp':      MLPModel,
}


def build_model(model_type: str, all_classes: list, **kwargs):
    """Instantiate a model wrapper by type string."""
    cls = MODEL_REGISTRY[model_type]
    model = cls(**kwargs)
    model.all_classes = list(all_classes)
    return model
