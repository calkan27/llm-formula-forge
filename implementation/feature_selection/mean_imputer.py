from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class MeanImputer:
	"""
	Train-only mean imputer.

	• fit(X): column means ignoring NaN (all-NaN -> 0.0 deterministically)
	• transform(X): replace NaN with stored means
	"""
	mean_: np.ndarray | None = None

	def fit(self, X: np.ndarray) -> "MeanImputer":
		X = np.asarray(X, dtype=np.float64)
		m = np.nanmean(X, axis=0)
		m = np.where(np.isfinite(m), m, 0.0)
		self.mean_ = m.astype(np.float64, copy=True)
		return self

	def transform(self, X: np.ndarray) -> np.ndarray:
		assert self.mean_ is not None
		X = np.asarray(X, dtype=np.float64)
		M = np.broadcast_to(self.mean_, X.shape)
		return np.where(np.isnan(X), M, X)

	def fit_transform(self, X: np.ndarray) -> np.ndarray:
		return self.fit(X).transform(X)

