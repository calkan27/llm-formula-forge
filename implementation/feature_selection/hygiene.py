from __future__ import annotations
import numpy as np
from .mean_imputer import MeanImputer
from .standard_scaler import StandardScaler


def train_only_fit_apply(
	X_train: np.ndarray,
	X_val: np.ndarray,
	X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MeanImputer, StandardScaler]:
	"""
	Compose MeanImputer -> StandardScaler with strict train-only fitting.
	Fit on train; apply to train/val/test using the same fitted objects.
	"""
	imp = MeanImputer().fit(X_train)
	Xtr_i = imp.transform(X_train)
	Xva_i = imp.transform(X_val)
	Xte_i = imp.transform(X_test)

	sc = StandardScaler().fit(Xtr_i)
	Xtr = sc.transform(Xtr_i)
	Xva = sc.transform(Xva_i)
	Xte = sc.transform(Xte_i)
	return Xtr, Xva, Xte, imp, sc

