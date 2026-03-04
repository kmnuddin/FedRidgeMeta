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

    def fit(self, Z, y):
        self._clf.fit(Z, y)
        if self._use_calib:
            try:
                self.calibrator = CalibratedClassifierCV(self._clf, cv='prefit', method='sigmoid')
                self.calibrator.fit(Z, y)
            except Exception:
                self.calibrator = None

    def predict_proba(self, Z):
        if self.calibrator is not None:
            return self.calibrator.predict_proba(Z)
        return self._clf.predict_proba(Z)
