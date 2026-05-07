import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

class RFWrapper:
    def __init__(self, n_estimators=100, class_weight='balanced_subsample',
                 max_depth=None, max_features='sqrt', n_jobs=-1, calibrate=False):
        self._clf = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight=class_weight,
            max_depth=max_depth,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=42,
        )
        self.calibrator = None
        self._use_calib = calibrate
        self.all_classes = None  # set externally to the full class list

    def fit(self, Z, y):
        self._clf.fit(Z, y)
        if self._use_calib:
            try:
                self.calibrator = CalibratedClassifierCV(self._clf, cv='prefit', method='sigmoid')
                self.calibrator.fit(Z, y)
            except Exception:
                self.calibrator = None

    def _pad_proba(self, P):
        """Pad predict_proba output to cover all_classes if the RF was
        trained on a subset (common under extreme non-IID).
        
        The forest's classes_ are the original string labels in sorted order.
        If the RF only saw a subset, we map by matching strings."""
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

    def predict_proba(self, Z):
        if self.calibrator is not None:
            P = self.calibrator.predict_proba(Z)
        else:
            P = self._clf.predict_proba(Z)
        return self._pad_proba(P)
